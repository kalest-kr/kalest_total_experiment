"""rng.py -- 이름이 고정된 독립 난수 스트림.

명세 10절: "각 실험의 데이터 생성·초기 배선·가중치·입력 잡음·학습 순서·진단
측정은 이름이 고정된 독립 RNG 스트림을 사용하라. 새 조건 하나를 추가했다고
기존 조건의 초기 상태가 바뀌면 안 된다."

구현: ``np.random.SeedSequence([master_seed, offset, sub_index])`` 로 만든다.
``offset`` 은 설정의 ``seeds.stream_offsets`` 에 **고정 정수**로 선언되어 있으므로
Python 의 실행별 ``hash()`` 에 의존하지 않는다. 스트림을 새로 추가해도 기존
스트림의 난수열은 바뀌지 않는다.

각 스트림의 ``bit_generator.state`` 는 체크포인트에 저장/복원되며, 진단 측정은
전용 ``diagnostics`` 스트림만 사용해 모델 스트림의 상태를 건드리지 않는다.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RngStreams:
    """이름 -> Generator 사전. 이름은 설정에서 고정 정수로 매핑된다."""

    def __init__(self, master_seed: int, stream_offsets: dict[str, int]) -> None:
        self.master_seed = int(master_seed)
        self.stream_offsets = {str(k): int(v) for k, v in stream_offsets.items()}
        self._gens: dict[str, np.random.Generator] = {}

    def get(self, name: str, sub_index: int = 0) -> np.random.Generator:
        """스트림을 얻는다. 같은 (name, sub_index) 는 같은 Generator 객체다.

        ``sub_index`` 는 같은 역할의 스트림을 표본/조건별로 분리할 때 쓴다
        (예: 자극 표본 i 의 입력 잡음).
        """
        if name not in self.stream_offsets:
            raise KeyError(
                f"등록되지 않은 RNG 스트림: {name!r}. "
                f"설정 seeds.stream_offsets 에 고정 정수로 추가해야 한다. "
                f"등록된 스트림: {sorted(self.stream_offsets)}"
            )
        key = f"{name}#{int(sub_index)}"
        gen = self._gens.get(key)
        if gen is None:
            ss = np.random.SeedSequence(
                [self.master_seed, self.stream_offsets[name], int(sub_index)]
            )
            gen = np.random.default_rng(ss)
            self._gens[key] = gen
        return gen

    def fresh(self, name: str, sub_index: int = 0) -> np.random.Generator:
        """캐시를 쓰지 않고 새로 만든 Generator (읽기 전용 진단용)."""
        if name not in self.stream_offsets:
            raise KeyError(f"등록되지 않은 RNG 스트림: {name!r}")
        ss = np.random.SeedSequence(
            [self.master_seed, self.stream_offsets[name], int(sub_index)]
        )
        return np.random.default_rng(ss)

    # --- 체크포인트 ---------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """모든 활성 스트림의 bit generator 상태 (JSON 직렬화 가능)."""
        return {
            "master_seed": self.master_seed,
            "stream_offsets": dict(self.stream_offsets),
            "states": {k: _jsonable_state(g.bit_generator.state)
                       for k, g in self._gens.items()},
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """저장된 상태 복원. 저장 시점 이후의 난수열을 그대로 이어간다."""
        self.master_seed = int(d["master_seed"])
        self.stream_offsets = {str(k): int(v) for k, v in d["stream_offsets"].items()}
        self._gens = {}
        for key, st in d["states"].items():
            name, _, sub = key.partition("#")
            gen = self.fresh(name, int(sub or 0))
            gen.bit_generator.state = _restore_state(st)
            self._gens[key] = gen

    def active_streams(self) -> list[str]:
        return sorted(self._gens)


def _jsonable_state(state: dict[str, Any]) -> dict[str, Any]:
    """numpy 정수를 파이썬 int 로 바꿔 JSON 저장 가능하게 만든다."""
    out: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out[k] = _jsonable_state(v)
        elif isinstance(v, np.ndarray):
            out[k] = {"__ndarray__": v.tolist(), "dtype": str(v.dtype)}
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _restore_state(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, dict) and "__ndarray__" in v:
            out[k] = np.asarray(v["__ndarray__"], dtype=np.dtype(v["dtype"]))
        elif isinstance(v, dict):
            out[k] = _restore_state(v)
        else:
            out[k] = v
    return out


def from_config(cfg: dict[str, Any]) -> RngStreams:
    return RngStreams(cfg["seeds"]["master"], cfg["seeds"]["stream_offsets"])


__all__ = ["RngStreams", "from_config"]
