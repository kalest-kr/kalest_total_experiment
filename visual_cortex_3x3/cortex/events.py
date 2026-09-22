"""events.py -- 입력 이벤트 로그와 지연 사건 큐.

명세 3절의 필드를 모두 갖춘다::

    event_id, parent_spike_id, sample_id, episode_id, src_id, dst_id,
    synapse_id, emit_time, arrival_time, arrival_step, weight_snapshot,
    source_gain_snapshot, target_compartment, event_type, amount, amount_unit

설계 선택
---------
* 저장은 **열 단위 NumPy 배열**(Structure-of-Arrays)이고 용량이 차면 2배로
  늘린다. 뉴런마다 무한 길이 Python 리스트를 들지 않는다.
* ``amount_unit`` 은 행마다 문자열을 저장하지 않고 정수 코드로 저장한다
  (:data:`AMOUNT_UNITS`). 문자열 매핑은 manifest 에 남는다.
* 뉴런별 조회는 ``dst_id`` 로 만든 **CSR 인덱스**를 지연 생성(lazy)한다.
  새 이벤트가 추가되면 인덱스를 무효화하고 다음 조회에서 다시 만든다.
* ``full`` / ``selected`` / ``summary`` 기록 모드를 지원한다. ``selected`` 와
  ``summary`` 는 선택 기준을 반드시 받아 manifest 에 남긴다. **조용히 버리고
  full 이라고 표시하지 않는다.**
* 용량 한도를 넘으면 :class:`CapacityExceeded` 를 올린다. 조용히 덮어쓰거나
  잘라내지 않는다.

발화 후에도 로그는 삭제하지 않는다. 현재 연산에 필요한 값은
:class:`cortex.records.NeuronArrays` 의 작업 버퍼(전도도/구간 누적)에 있고
로그와 분리되어 있다.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ids import COMPARTMENT_NAMES, EVENT_TYPES

#: amount 의 단위 코드. sum_threshold 모드는 무차원 기여량, 전도도 모드는 nS 다.
AMOUNT_UNITS: tuple[str, ...] = ("dimensionless", "nS", "pA", "spike", "weight_delta", "mV")
AMOUNT_UNIT_INDEX: dict[str, int] = {n: i for i, n in enumerate(AMOUNT_UNITS)}

EVENT_TYPE_INDEX: dict[str, int] = {n: i for i, n in enumerate(EVENT_TYPES)}

#: 이벤트 1행의 대략적 바이트 수 (용량 추정용, DATA_SCHEMA.md 와 일치)
BYTES_PER_EVENT_ROW: int = 88


class CapacityExceeded(RuntimeError):
    """설정한 이벤트/상태 용량 한도를 넘었을 때. 실행을 명시적으로 중단한다."""


class _Growable:
    """용량 2배 증가 방식의 1차원 growable 배열."""

    __slots__ = ("_buf", "_n")

    def __init__(self, dtype: Any, capacity: int = 1024) -> None:
        self._buf = np.zeros(int(capacity), dtype=dtype)
        self._n = 0

    def append_many(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=self._buf.dtype).ravel()
        need = self._n + v.size
        if need > self._buf.size:
            cap = max(need, self._buf.size * 2)
            new = np.zeros(cap, dtype=self._buf.dtype)
            new[: self._n] = self._buf[: self._n]
            self._buf = new
        self._buf[self._n : need] = v
        self._n = need

    @property
    def data(self) -> np.ndarray:
        """현재 유효 구간의 뷰 (사본 아님)."""
        return self._buf[: self._n]

    def __len__(self) -> int:
        return self._n

    def truncate(self, n: int) -> None:
        self._n = int(n)


class EventLog:
    """장기 입력/발화/학습 이벤트 로그.

    Parameters
    ----------
    n_neurons : int
    mode : {"full", "selected", "summary"}
    selected_neurons : array-like[int] | None
        ``selected`` 모드에서 기록할 뉴런 ID.
    selection_criterion : str
        ``full`` 이 아닐 때 필수. manifest 에 저장되어 재현 범위를 명시한다.
    max_events : int
        넘으면 :class:`CapacityExceeded`.
    """

    COLUMNS: tuple[str, ...] = (
        "event_id", "parent_spike_id", "sample_id", "episode_id",
        "src_id", "dst_id", "synapse_id",
        "emit_time_ms", "arrival_time_ms", "arrival_step",
        "weight_snapshot", "source_gain_snapshot",
        "target_compartment", "event_type", "amount", "amount_unit",
    )

    def __init__(self, n_neurons: int, mode: str = "full",
                 selected_neurons: Any = None, selection_criterion: str = "",
                 max_events: int = 20_000_000, capacity: int = 4096) -> None:
        if mode not in ("full", "selected", "summary"):
            raise ValueError("mode 는 full/selected/summary")
        if mode != "full" and not selection_criterion:
            raise ValueError(
                "full 이 아닌 기록 모드는 selection_criterion 을 반드시 지정해야 한다 "
                "(무엇을 기록하지 않는지 manifest 에 남긴다)"
            )
        self.n_neurons = int(n_neurons)
        self.mode = mode
        self.selection_criterion = selection_criterion
        self.max_events = int(max_events)

        self._keep = np.ones(self.n_neurons, dtype=bool)
        if mode == "selected":
            self._keep[:] = False
            if selected_neurons is not None:
                self._keep[np.asarray(selected_neurons, dtype=np.int64)] = True
        elif mode == "summary":
            self._keep[:] = False

        self._cols: dict[str, _Growable] = {
            "event_id": _Growable(np.int64, capacity),
            "parent_spike_id": _Growable(np.int64, capacity),
            "sample_id": _Growable(np.int32, capacity),
            "episode_id": _Growable(np.int32, capacity),
            "src_id": _Growable(np.int32, capacity),
            "dst_id": _Growable(np.int32, capacity),
            "synapse_id": _Growable(np.int64, capacity),
            "emit_time_ms": _Growable(np.float64, capacity),
            "arrival_time_ms": _Growable(np.float64, capacity),
            "arrival_step": _Growable(np.int64, capacity),
            "weight_snapshot": _Growable(np.float64, capacity),
            "source_gain_snapshot": _Growable(np.float64, capacity),
            "target_compartment": _Growable(np.int8, capacity),
            "event_type": _Growable(np.int8, capacity),
            "amount": _Growable(np.float64, capacity),
            "amount_unit": _Growable(np.int8, capacity),
        }
        self._index_ptr: np.ndarray | None = None
        self._index_rows: np.ndarray | None = None
        self._index_valid_upto = 0

        #: 기록 모드 때문에 저장하지 않은 이벤트 수 (조용히 버리지 않기 위해 센다)
        self.n_skipped_by_mode = 0
        #: summary 모드에서 유지하는 집계
        self.summary_counts: dict[int, int] = {}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._cols["event_id"])

    @property
    def n_events(self) -> int:
        return len(self._cols["event_id"])

    def column(self, name: str) -> np.ndarray:
        return self._cols[name].data

    def append(self, *, event_id: np.ndarray, parent_spike_id: np.ndarray,
               sample_id: int, episode_id: int,
               src_id: np.ndarray, dst_id: np.ndarray, synapse_id: np.ndarray,
               emit_time_ms: np.ndarray, arrival_time_ms: np.ndarray,
               arrival_step: np.ndarray, weight_snapshot: np.ndarray,
               source_gain_snapshot: np.ndarray, target_compartment: np.ndarray,
               event_type: str, amount: np.ndarray, amount_unit: str) -> int:
        """이벤트 여러 건을 한 번에 기록한다 (부작용: 로그 확장).

        Returns
        -------
        int : 실제로 저장한 행 수 (기록 모드로 걸러진 건 제외).
        """
        et = EVENT_TYPE_INDEX[event_type]
        au = AMOUNT_UNIT_INDEX[amount_unit]
        dst = np.asarray(dst_id, dtype=np.int64)

        if self.mode == "summary":
            self.summary_counts[et] = self.summary_counts.get(et, 0) + int(dst.size)
            self.n_skipped_by_mode += int(dst.size)
            return 0

        if self.mode == "selected":
            keep = np.zeros(dst.size, dtype=bool)
            valid = (dst >= 0) & (dst < self.n_neurons)
            keep[valid] = self._keep[dst[valid]]
            # 발화 이벤트는 src_id 로 판단한다 (dst 가 -1)
            src = np.asarray(src_id, dtype=np.int64)
            is_spike = dst < 0
            if is_spike.any():
                sv = is_spike & (src >= 0) & (src < self.n_neurons)
                keep[sv] = self._keep[src[sv]]
            n_drop = int((~keep).sum())
            if n_drop:
                self.n_skipped_by_mode += n_drop
            if not keep.any():
                return 0
            sel = np.nonzero(keep)[0]
        else:
            sel = None

        def pick(arr: Any, dtype: Any) -> np.ndarray:
            a = np.asarray(arr, dtype=dtype)
            if a.ndim == 0:
                a = np.full(dst.size, a, dtype=dtype)
            return a if sel is None else a[sel]

        n_new = dst.size if sel is None else sel.size
        if self.n_events + n_new > self.max_events:
            raise CapacityExceeded(
                f"이벤트 한도 초과: 현재 {self.n_events} + 신규 {n_new} > "
                f"max_events={self.max_events}. 설정 recording.max_events 를 늘리거나 "
                f"recording.mode 를 selected/summary 로 바꾸고 selection_criterion 을 기록하라."
            )

        self._cols["event_id"].append_many(pick(event_id, np.int64))
        self._cols["parent_spike_id"].append_many(pick(parent_spike_id, np.int64))
        self._cols["sample_id"].append_many(np.full(n_new, int(sample_id), np.int32))
        self._cols["episode_id"].append_many(np.full(n_new, int(episode_id), np.int32))
        self._cols["src_id"].append_many(pick(src_id, np.int32))
        self._cols["dst_id"].append_many(pick(dst_id, np.int32))
        self._cols["synapse_id"].append_many(pick(synapse_id, np.int64))
        self._cols["emit_time_ms"].append_many(pick(emit_time_ms, np.float64))
        self._cols["arrival_time_ms"].append_many(pick(arrival_time_ms, np.float64))
        self._cols["arrival_step"].append_many(pick(arrival_step, np.int64))
        self._cols["weight_snapshot"].append_many(pick(weight_snapshot, np.float64))
        self._cols["source_gain_snapshot"].append_many(pick(source_gain_snapshot, np.float64))
        self._cols["target_compartment"].append_many(pick(target_compartment, np.int8))
        self._cols["event_type"].append_many(np.full(n_new, et, np.int8))
        self._cols["amount"].append_many(pick(amount, np.float64))
        self._cols["amount_unit"].append_many(np.full(n_new, au, np.int8))
        self._index_ptr = None  # 인덱스 무효화
        return int(n_new)

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """dst_id 기준 CSR 인덱스를 만든다 (조회 시 지연 생성)."""
        dst = self._cols["dst_id"].data
        n = self.n_neurons
        valid = (dst >= 0) & (dst < n)
        rows = np.nonzero(valid)[0]
        keys = dst[rows].astype(np.int64)
        order = np.argsort(keys, kind="stable")
        self._index_rows = rows[order]
        counts = np.bincount(keys, minlength=n)
        ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=ptr[1:])
        self._index_ptr = ptr
        self._index_valid_upto = self.n_events

    def count_for_neuron(self, neuron_id: int) -> int:
        if self._index_ptr is None or self._index_valid_upto != self.n_events:
            self._build_index()
        assert self._index_ptr is not None
        i = int(neuron_id)
        return int(self._index_ptr[i + 1] - self._index_ptr[i])

    def query_neuron(self, neuron_id: int, t_from_ms: float | None = None,
                     t_to_ms: float | None = None,
                     limit: int | None = None) -> dict[str, np.ndarray]:
        """한 뉴런의 도착 이벤트를 시간 구간으로 조회한다 (부작용 없음)."""
        if self._index_ptr is None or self._index_valid_upto != self.n_events:
            self._build_index()
        assert self._index_ptr is not None and self._index_rows is not None
        i = int(neuron_id)
        lo, hi = int(self._index_ptr[i]), int(self._index_ptr[i + 1])
        rows = self._index_rows[lo:hi]
        if rows.size and (t_from_ms is not None or t_to_ms is not None):
            t = self._cols["arrival_time_ms"].data[rows]
            m = np.ones(rows.size, dtype=bool)
            if t_from_ms is not None:
                m &= t >= float(t_from_ms)
            if t_to_ms is not None:
                m &= t <= float(t_to_ms)
            rows = rows[m]
        if limit is not None and rows.size > limit:
            rows = rows[: int(limit)]
        return {name: col.data[rows] for name, col in self._cols.items()}

    def query_rows(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        rows = np.asarray(rows, dtype=np.int64)
        return {name: col.data[rows] for name, col in self._cols.items()}

    def all_columns(self) -> dict[str, np.ndarray]:
        return {name: col.data for name, col in self._cols.items()}

    def schema(self) -> dict[str, Any]:
        return {
            "columns": list(self.COLUMNS),
            "dtypes": {k: str(v.data.dtype) for k, v in self._cols.items()},
            "event_types": list(EVENT_TYPES),
            "amount_units": list(AMOUNT_UNITS),
            "compartments": list(COMPARTMENT_NAMES),
            "mode": self.mode,
            "selection_criterion": self.selection_criterion,
            "n_skipped_by_mode": self.n_skipped_by_mode,
        }


class DelayQueue:
    """도착 스텝별 버킷 큐.

    ``schedule(step, ...)`` 로 예약하고 ``pop(step)`` 으로 꺼낸다. 꺼낸 버킷은
    비워지므로 같은 사건이 두 번 주입되지 않는다.

    지연 0 은 허용하지 않는다 (``min_delay_steps >= 1``). 0 지연 재귀로 한
    스텝 안에서 무한 루프가 생기는 것을 구조적으로 막는다.
    """

    def __init__(self) -> None:
        self._buckets: dict[int, list[dict[str, np.ndarray]]] = {}
        self.n_scheduled = 0

    def schedule(self, arrival_step: int, payload: dict[str, np.ndarray]) -> None:
        step = int(arrival_step)
        if step < 0:
            raise ValueError("arrival_step 은 0 이상이어야 한다")
        self._buckets.setdefault(step, []).append(payload)
        first = next(iter(payload.values()))
        self.n_scheduled += int(np.asarray(first).size)

    def pop(self, step: int) -> dict[str, np.ndarray]:
        """해당 스텝의 사건을 합쳐 돌려주고 버킷을 비운다."""
        parts = self._buckets.pop(int(step), None)
        if not parts:
            return {}
        keys = parts[0].keys()
        return {k: np.concatenate([p[k] for p in parts]) for k in keys}

    def peek_steps(self) -> list[int]:
        return sorted(self._buckets)

    def pending_count(self) -> int:
        total = 0
        for parts in self._buckets.values():
            for p in parts:
                total += int(np.asarray(next(iter(p.values()))).size)
        return total

    def clear(self) -> None:
        self._buckets.clear()

    # --- 체크포인트 ---------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"steps": [], "payloads": []}
        for step in sorted(self._buckets):
            merged = {}
            keys = self._buckets[step][0].keys()
            for k in keys:
                merged[k] = np.concatenate([p[k] for p in self._buckets[step]])
            out["steps"].append(int(step))
            out["payloads"].append({k: v for k, v in merged.items()})
        return out

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._buckets.clear()
        self.n_scheduled = 0
        for step, payload in zip(d["steps"], d["payloads"]):
            self.schedule(int(step), {k: np.asarray(v) for k, v in payload.items()})


def quantize_delay(delay_ms: float | np.ndarray, dt_ms: float,
                   min_steps: int = 1, rounding: str = "ceil") -> np.ndarray:
    """연속 지연(ms)을 dt 격자 스텝 수로 양자화한다.

    Parameters
    ----------
    delay_ms : float | ndarray
    dt_ms : float
    min_steps : int
        최소 지연 스텝. 1 이상이어야 0 지연 재귀가 생기지 않는다.
    rounding : {"ceil", "round"}

    Returns
    -------
    ndarray[int64] : 양자화된 스텝 수 (>= min_steps)

    연속 지연과 양자화 지연을 **둘 다** 기록한다 (명세 6절).
    """
    if min_steps < 1:
        raise ValueError("min_steps 는 1 이상이어야 한다 (0 지연 재귀 방지)")
    d = np.asarray(delay_ms, dtype=np.float64) / float(dt_ms)
    if rounding == "ceil":
        steps = np.ceil(d)
    elif rounding == "round":
        steps = np.rint(d)
    else:
        raise ValueError("rounding 은 'ceil' 또는 'round'")
    return np.maximum(steps.astype(np.int64), int(min_steps))


__all__ = [
    "AMOUNT_UNITS", "AMOUNT_UNIT_INDEX", "EVENT_TYPE_INDEX", "BYTES_PER_EVENT_ROW",
    "CapacityExceeded", "EventLog", "DelayQueue", "quantize_delay",
]
