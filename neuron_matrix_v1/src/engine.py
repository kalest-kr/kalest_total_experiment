"""engine.py -- 동기식 이산 시점 실행 엔진.

명세 [4], [5]. 한 tick 의 계산은 정확히 다음 순서다.

1. 현재 tick 의 도착 이벤트와 외부 자극을 수집한다.
2. 연결별 도착값 ``s_k`` 를 정리한다.
3. 모든 뉴런의 E, I, b, u 를 계산한다.
4. **같은 가중치 상태**에서 모든 뉴런의 q 를 결정한다.
5. 발화한 뉴런의 출력을 각 연결의 delay_ticks 에 따라 미래 큐에 넣는다.
6. 관찰 시점의 상태/연결 기여 스냅샷을 저장한다.
7. tick 을 증가시킨다.

한 tick 안에서 연쇄 발화는 일어나지 않는다. 이 모델은 막전위 적분 모델이
아니며, 입력이 없는 다음 tick 에는 E=I=0 으로 다시 계산된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .connections import ConnectionTable
from .events import ExternalDrive, ExternalDriveBuffer, InputBuffer, InputEvent
from .neuron import Population


@dataclass
class TrialResult:
    """``run_trial`` 반환값. 가중치를 바꾸지 않는다.

    Attributes
    ----------
    trial_id : int | str
    observed_q : np.ndarray, shape (N,) int8
        각 뉴런의 ``observation_tick`` 에서 읽은 발화 (0/1).
        해당 tick 이 실행 범위 밖이면 0 이다.
    observed_u, observed_E, observed_I, observed_b : np.ndarray, shape (N,) float64
        같은 관찰 시점의 상태 값.
    arrivals_by_tick : dict[int, np.ndarray]
        관찰 tick -> ``(K,)`` 연결별 도착값 ``s_k`` 사본.
        국소 학습의 ``e_k`` 는 여기서 온다.
    n_ticks : int
        실행한 tick 수.
    n_events : int
        생성/수신한 연결 이벤트 수 (외부 자극 제외).
    n_external_drives : int
    weight_version : int
        이 시행에 사용한 가중치 버전.
    """

    trial_id: Any
    observed_q: np.ndarray
    observed_u: np.ndarray
    observed_E: np.ndarray
    observed_I: np.ndarray
    observed_b: np.ndarray
    arrivals_by_tick: dict[int, np.ndarray]
    n_ticks: int
    n_events: int
    n_external_drives: int
    weight_version: int
    tick_history: list[dict[str, Any]] = field(default_factory=list)

    def arrivals_at(self, tick: int) -> np.ndarray:
        return self.arrivals_by_tick[int(tick)]


class SimulationEngine:
    """동기식 실행 엔진.

    Parameters
    ----------
    population : Population
    table : ConnectionTable
        ``finalize()`` 된 상태여야 한다.
    n_ticks : int
        한 시행에서 실행할 tick 수 (0..n_ticks-1).
    event_mode : {"bulk", "object"}
        ``"object"`` 이면 전송 시 ``InputEvent`` 객체를 1건씩 만들어
        ``InputBuffer.receive`` 로 검증/저장한다 (느린 참조 경로).
        ``"bulk"`` 는 동일 계산의 벡터화 경로다. 두 경로의 결과가 같아야 한다.
    recorder : TraceRecorder | None
        None 이면 진단 기록을 하지 않는다. 진단은 계산에 영향을 주지 않는다.
    """

    def __init__(
        self,
        population: Population,
        table: ConnectionTable,
        n_ticks: int = 4,
        event_mode: str = "bulk",
        recorder: Any = None,
    ) -> None:
        if event_mode not in ("bulk", "object"):
            raise ValueError("event_mode 는 'bulk' 또는 'object' 여야 한다")
        self.pop = population
        self.table = table
        self.n_ticks = int(n_ticks)
        self.event_mode = event_mode
        self.recorder = recorder

        self.tick = 0
        self.buffer = InputBuffer(table, record_events=(event_mode == "object"))
        self.drives = ExternalDriveBuffer(len(population))

        self._n = len(population)
        self._b_base = population.b_base_vector()          # (N,) 뷰
        self._obs_tick = population.observation_tick_vector()  # (N,) 뷰
        self._correction = population.store.correction_input   # (N,) 뷰
        self._fired = population.store.fired                   # (N,) 뷰
        self._ext_state = population.store.external_drive      # (N,) 뷰
        self._correction_tick = self._obs_tick.copy()

        # 초기 파라미터 스냅샷 (reset_parameters 용)
        self._initial_states = population.store.states.copy()
        self._initial_b_base = population.store.b_base.copy()
        self._initial_weights = table.snapshot_weights()
        self._initial_weight_version = table.weight_version

        self._exc = table.exc_order
        self._inh = table.inh_order
        self._post = table.post_id
        self._pre = table.pre_id

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------
    def reset_transient_state(self) -> None:
        """버퍼, 이벤트 큐, E/I/u, 발화, 교정 입력, tick 을 초기화한다.

        **학습된 연결 가중치와 고정 메타데이터는 유지한다.** (명세 [5])
        초기화 후 b 는 b_base 로 돌아간다.
        """
        self.buffer.clear()
        self.drives.clear()
        self.pop.reset_transient()
        self._correction[:] = 0.0
        self.tick = 0

    def reset_parameters(self) -> None:
        """threshold / P / 위치 / b_base / 가중치를 **구성 직후 값**으로 되돌린다.

        ``reset_transient_state`` 와 달리 학습 결과를 파기한다.
        """
        self.pop.store.states[...] = self._initial_states
        self.pop.store.b_base[...] = self._initial_b_base
        self.table.restore_weights(self._initial_weights, self._initial_weight_version)
        self.reset_transient_state()

    # ------------------------------------------------------------------
    # 자극 주입
    # ------------------------------------------------------------------
    def inject_external_drive(self, tick: int, neuron_ids: np.ndarray, values: np.ndarray) -> None:
        """L4 등에 외부 감각값을 주입한다 (연결이 아님)."""
        self.drives.receive_vector(tick, neuron_ids, values)

    def inject_event(self, event: InputEvent) -> None:
        """연결 이벤트 1건을 직접 주입한다 (테스트/진단용)."""
        self.buffer.receive(event)

    def set_correction(self, correction: Mapping[int, float] | None) -> None:
        """관찰 시점에 적용할 교정 입력 c_i 를 설정한다.

        correction 이 None 이면 모두 0 이다. 교정은 b 의 성분으로만 들어가며
        external_sensory_drive 를 바꾸지 않는다 (명세 [4], [10]).
        """
        self._correction[:] = 0.0
        if correction is None:
            return
        if isinstance(correction, tuple):
            ids, vals = correction
            self._correction[np.asarray(ids, dtype=np.int64)] = np.asarray(
                vals, dtype=np.float64
            )
        else:
            for nid, c in correction.items():
                self._correction[int(nid)] = float(c)

    # ------------------------------------------------------------------
    # 한 tick
    # ------------------------------------------------------------------
    def step(self) -> dict[str, Any]:
        """명세 [5]의 7단계를 정확히 수행하고 tick 요약 dict 를 반환한다."""
        t = self.tick
        states = self.pop.store.states
        n = self._n

        # 1-2. 도착 이벤트 / 외부 자극 수집 및 연결별 정리
        s, events = self.buffer.pop_tick(t)
        ext = self.drives.pop_tick(t)

        # 3. E, I, b, u 계산 (합산 순서는 등록 순서와 무관한 정준 순열)
        w = self.table.weight
        exc, inh = self._exc, self._inh
        E = np.bincount(self._post[exc], weights=w[exc] * s[exc], minlength=n)
        I = np.bincount(self._post[inh], weights=w[inh] * s[inh], minlength=n)
        E = E + ext

        corr = np.where(self._correction_tick == t, self._correction, 0.0)
        b = self._b_base + corr
        self._ext_state[:] = ext
        u = E - I + b

        states[:, 2, 0] = E
        states[:, 2, 1] = I
        states[:, 2, 2] = b
        states[:, 1, 0] = u

        # 4. 같은 가중치 상태에서 일괄 발화 판정
        threshold = states[:, 1, 1]
        q = (u >= threshold).astype(np.int8)
        self._fired[:] = q   # Neuron.fired 는 이 배열의 property 다 (중복 저장 없음)

        # 5. 발화 뉴런의 출력을 지연에 따라 미래 큐에 넣는다 (q=0 은 생략)
        P = states[:, 1, 2]
        n_emitted = 0
        fired_mask = q.astype(bool)
        if fired_mask.any():
            for d, idx in self.table.delay_groups:
                active = idx[fired_mask[self._pre[idx]]]
                if active.size == 0:
                    continue
                values = P[self._pre[active]]
                if self.event_mode == "object":
                    for c, v in zip(active.tolist(), values.tolist()):
                        self.buffer.receive(
                            InputEvent(t + d, int(c), int(self._pre[c]), float(v))
                        )
                else:
                    self.buffer.receive_bulk(t + d, active, values)
                n_emitted += int(active.size)

        # 6. 관찰 시점 스냅샷
        obs_mask = self._obs_tick == t
        if self.recorder is not None:
            self.recorder.record_tick(
                engine=self,
                tick=t,
                s=s,
                ext=ext,
                E=E,
                I=I,
                b=b,
                u=u,
                q=q,
                corr=corr,
            )

        # 7. tick 증가
        self.tick = t + 1

        return {
            "tick": t,
            "n_arrivals": int(np.count_nonzero(s)),
            "n_emitted": n_emitted,
            "n_fired": int(q.sum()),
            "observed_ids": np.nonzero(obs_mask)[0],
            "s": s,
            "E": E,
            "I": I,
            "b": b,
            "u": u,
            "q": q,
            "events": events,
        }

    # ------------------------------------------------------------------
    # 한 시행
    # ------------------------------------------------------------------
    def run_trial(
        self,
        external_drives: Mapping[int, float] | tuple[np.ndarray, np.ndarray] | None = None,
        correction: Mapping[int, float] | None = None,
        trial_id: Any = 0,
        drive_tick: int = 0,
        extra_events: list[InputEvent] | None = None,
        snapshot_ticks: tuple[int, ...] | None = None,
        keep_tick_history: bool = False,
    ) -> TrialResult:
        """영상 1장(또는 1개 자극)의 자유/유도 실행. **가중치를 바꾸지 않는다.**

        Parameters
        ----------
        external_drives
            ``{neuron_id: value}`` 또는 ``(ids, values)`` 배열 쌍.
            ``drive_tick`` 에 주입한다. None 이면 주입 없음.
        correction
            ``{neuron_id: c}`` 또는 ``(ids, values)``.
            각 뉴런의 관찰 시점에만 b 에 더해진다.
        extra_events
            일시 상태 초기화 **후** 버퍼에 직접 넣을 ``InputEvent`` 목록
            (검사/진단용). 정상 회로 실행에서는 사용하지 않는다.
        snapshot_ticks
            연결별 도착값 ``s`` 사본을 남길 tick 목록.
            None 이면 모든 뉴런의 observation_tick 집합을 사용한다.
        keep_tick_history
            True 면 tick 별 요약을 ``TrialResult.tick_history`` 에 담는다
            (메모리 사용 증가, 기본 False).

        Returns
        -------
        TrialResult
        """
        self.reset_transient_state()
        if trial_id is not None and self.recorder is not None:
            self.recorder.begin_trial(trial_id)

        if external_drives is not None:
            if isinstance(external_drives, tuple):
                ids, vals = external_drives
            else:
                ids = np.fromiter(external_drives.keys(), dtype=np.int64)
                vals = np.fromiter(external_drives.values(), dtype=np.float64)
            self.inject_external_drive(drive_tick, ids, vals)
        if extra_events:
            for ev in extra_events:
                self.buffer.receive(ev)
        self.set_correction(correction)

        if snapshot_ticks is None:
            snapshot_ticks = tuple(sorted(set(int(x) for x in self._obs_tick.tolist())))
        snap_set = set(int(x) for x in snapshot_ticks)

        n = self._n
        obs_q = np.zeros(n, dtype=np.int8)
        obs_u = np.zeros(n, dtype=np.float64)
        obs_E = np.zeros(n, dtype=np.float64)
        obs_I = np.zeros(n, dtype=np.float64)
        obs_b = np.zeros(n, dtype=np.float64)
        arrivals: dict[int, np.ndarray] = {}
        history: list[dict[str, Any]] = []
        n_events = 0

        for _ in range(self.n_ticks):
            t = self.tick
            info = self.step()
            ids = info["observed_ids"]
            if ids.size:
                obs_q[ids] = info["q"][ids]
                obs_u[ids] = info["u"][ids]
                obs_E[ids] = info["E"][ids]
                obs_I[ids] = info["I"][ids]
                obs_b[ids] = info["b"][ids]
            if t in snap_set:
                arrivals[t] = info["s"].copy()
            n_events += info["n_emitted"]
            if keep_tick_history:
                history.append(
                    {
                        "tick": t,
                        "n_fired": info["n_fired"],
                        "n_arrivals": info["n_arrivals"],
                        "n_emitted": info["n_emitted"],
                    }
                )

        return TrialResult(
            trial_id=trial_id,
            observed_q=obs_q,
            observed_u=obs_u,
            observed_E=obs_E,
            observed_I=obs_I,
            observed_b=obs_b,
            arrivals_by_tick=arrivals,
            n_ticks=self.n_ticks,
            n_events=n_events,
            n_external_drives=self.drives.n_drives_received,
            weight_version=self.table.weight_version,
            tick_history=history,
        )

    # ------------------------------------------------------------------
    def state_snapshot(self) -> dict[str, np.ndarray]:
        """상태/가중치 사본 (진단 개입 전후 보존 확인용, 부작용 없음)."""
        return {
            "states": self.pop.store.states.copy(),
            "weights": self.table.snapshot_weights(),
            "weight_version": np.array([self.table.weight_version]),
            "tick": np.array([self.tick]),
            "correction": self._correction.copy(),
        }

    def restore_snapshot(self, snap: dict[str, np.ndarray]) -> None:
        self.pop.store.states[...] = snap["states"]
        self.table.restore_weights(snap["weights"], int(snap["weight_version"][0]))
        self.tick = int(snap["tick"][0])
        self._correction[...] = snap["correction"]
        self._fired[:] = 0
