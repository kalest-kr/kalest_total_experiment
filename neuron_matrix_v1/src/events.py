"""events.py -- 일시적인 입력 이벤트, 외부 자극, 수신 버퍼.

명세 [3]-B.

* ``InputEvent`` 는 **연결을 통해** 도착한 값이다. 수신 뉴런은
  ``connection_id`` 로 결정되며 ``source_id`` 는 연결 목록과 대조해 검증한다.
  ``received_value`` 는 가중치를 적용하기 **전**의 도착값이고, 일반 뉴런에서
  온 값에는 송신 뉴런의 P 가 이미 곱해져 있다.
* ``ExternalDrive`` 는 전처리기가 L4 에 직접 주입하는 연속값이다.
  신경 연결이 아니므로 ConnectionTable 을 거치지 않는다.
* ``InputBuffer.receive`` 는 **저장만** 한다. E/I 누적이나 발화를 하지 않는다.
* 처리한 tick 의 버퍼는 ``pop_tick`` 으로 꺼내며 그 즉시 비워진다.
  같은 이벤트를 두 번 처리하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connections import ConnectionTable


@dataclass(frozen=True)
class InputEvent:
    """연결을 통해 도착한 1건의 입력.

    Attributes
    ----------
    arrival_tick : int
        도착 시점 (이산 계산 단위). 학습 시행 번호와 무관하다.
    connection_id : int
        도착한 연결. 수신 뉴런은 이것으로 결정된다.
    source_id : int
        송신 뉴런 ID. 연결 목록과 일치해야 한다.
    received_value : float
        가중치 적용 **전**의 비음수 도착값 (송신 P 는 이미 포함).
    """

    arrival_tick: int
    connection_id: int
    source_id: int
    received_value: float


@dataclass(frozen=True)
class ExternalDrive:
    """전처리기가 뉴런에 직접 주입하는 비음수 감각값 (연결이 아님).

    Attributes
    ----------
    tick : int
    neuron_id : int
    value : float
        비음수. L4 의 ``external_sensory_drive`` 로 들어가며 E 에 단위 이득으로
        더해진다. 교사 교정은 이 항을 절대 바꾸지 않는다 (명세 [4]).
    """

    tick: int
    neuron_id: int
    value: float


class InputBuffer:
    """tick 별 연결 도착값 누적 버퍼.

    내부적으로는 tick -> ``(K,) float64`` 배열(연결별 합 ``s_k``)을 쓴다.
    ``record_events=True`` 이면 동일 tick 의 ``InputEvent`` 객체 목록도
    함께 보관한다 (진단/검증용). 두 경로는 같은 누적값을 쓰므로 결과가
    같아야 하며 이는 T3/T4 에서 검사한다.
    """

    def __init__(self, table: ConnectionTable, record_events: bool = False) -> None:
        self._table = table
        self._k = table.n_connections
        self._pending: dict[int, np.ndarray] = {}
        self._events: dict[int, list[InputEvent]] = {}
        self.record_events = bool(record_events)
        self.n_events_received = 0

    # --- 저장만 하는 수신 함수 ---------------------------------------------
    def receive(self, event: InputEvent) -> None:
        """이벤트 1건을 저장한다. E/I 누적이나 발화를 하지 않는다 (부작용: 버퍼)."""
        self._table.validate_event_source(event.connection_id, event.source_id)
        if event.received_value < 0.0:
            raise ValueError("received_value 는 비음수여야 한다")
        if event.arrival_tick < 0:
            raise ValueError("arrival_tick 은 0 이상이어야 한다")
        arr = self._pending.get(event.arrival_tick)
        if arr is None:
            arr = np.zeros(self._k, dtype=np.float64)
            self._pending[event.arrival_tick] = arr
        arr[event.connection_id] += float(event.received_value)
        if self.record_events:
            self._events.setdefault(event.arrival_tick, []).append(event)
        self.n_events_received += 1

    def receive_bulk(
        self, arrival_tick: int, connection_ids: np.ndarray, values: np.ndarray
    ) -> None:
        """벡터화 수신. ``receive`` 와 동일한 검증/누적을 배열 단위로 수행한다.

        connection_ids 는 오름차순이어야 한다 (누적 순서 결정성 보장).
        """
        cids = np.asarray(connection_ids, dtype=np.int64)
        vals = np.asarray(values, dtype=np.float64)
        if cids.size == 0:
            return
        if np.any(vals < 0.0):
            raise ValueError("received_value 는 비음수여야 한다")
        arr = self._pending.get(int(arrival_tick))
        if arr is None:
            arr = np.zeros(self._k, dtype=np.float64)
            self._pending[int(arrival_tick)] = arr
        np.add.at(arr, cids, vals)
        if self.record_events:
            bucket = self._events.setdefault(int(arrival_tick), [])
            pre = self._table.pre_id
            for c, v in zip(cids.tolist(), vals.tolist()):
                bucket.append(InputEvent(int(arrival_tick), int(c), int(pre[c]), float(v)))
        self.n_events_received += int(cids.size)

    # --- 처리 -------------------------------------------------------------
    def pop_tick(self, tick: int) -> tuple[np.ndarray, list[InputEvent]]:
        """해당 tick 의 연결별 도착합 ``s`` 와 (기록된 경우) 이벤트 목록을 꺼낸다.

        Returns
        -------
        s : np.ndarray, shape (K,)
            연결별 도착값 합. 도착이 없으면 0.
        events : list[InputEvent]
            ``record_events=False`` 이면 빈 목록.

        부작용: 해당 tick 의 버퍼를 비운다 (같은 이벤트 재처리 방지).
        """
        s = self._pending.pop(int(tick), None)
        if s is None:
            s = np.zeros(self._k, dtype=np.float64)
        ev = self._events.pop(int(tick), [])
        return s, ev

    def clear(self) -> None:
        """모든 tick 의 버퍼와 이벤트를 삭제한다 (영상 사이 초기화)."""
        self._pending.clear()
        self._events.clear()
        self.n_events_received = 0

    def pending_ticks(self) -> list[int]:
        return sorted(self._pending.keys())

    def is_empty(self) -> bool:
        return not self._pending and not self._events


class ExternalDriveBuffer:
    """tick 별 외부 감각 주입 버퍼 (연결과 완전히 분리).

    같은 (tick, neuron) 에 여러 번 주입하면 합산된다.
    """

    def __init__(self, n_neurons: int) -> None:
        self._n = int(n_neurons)
        self._pending: dict[int, np.ndarray] = {}
        self.n_drives_received = 0

    def receive(self, drive: ExternalDrive) -> None:
        if drive.value < 0.0:
            raise ValueError("external_sensory_drive 는 비음수여야 한다")
        arr = self._pending.get(drive.tick)
        if arr is None:
            arr = np.zeros(self._n, dtype=np.float64)
            self._pending[drive.tick] = arr
        arr[drive.neuron_id] += float(drive.value)
        self.n_drives_received += 1

    def receive_vector(self, tick: int, neuron_ids: np.ndarray, values: np.ndarray) -> None:
        vals = np.asarray(values, dtype=np.float64)
        if np.any(vals < 0.0):
            raise ValueError("external_sensory_drive 는 비음수여야 한다")
        arr = self._pending.get(int(tick))
        if arr is None:
            arr = np.zeros(self._n, dtype=np.float64)
            self._pending[int(tick)] = arr
        np.add.at(arr, np.asarray(neuron_ids, dtype=np.int64), vals)
        self.n_drives_received += int(np.asarray(neuron_ids).size)

    def pop_tick(self, tick: int) -> np.ndarray:
        arr = self._pending.pop(int(tick), None)
        if arr is None:
            arr = np.zeros(self._n, dtype=np.float64)
        return arr

    def clear(self) -> None:
        self._pending.clear()
        self.n_drives_received = 0

    def is_empty(self) -> bool:
        return not self._pending
