"""rngs.py -- 이름이 붙은 독립 난수 스트림.

명세 [13]: "스트림 이름을 고정 정수에 매핑해 SeedSequence 로 생성한다.
Python 의 실행별 hash 값에 의존하지 않는다."

각 스트림은 ``np.random.SeedSequence([seed, STREAM_IDS[name]])`` 에서
만들어지므로, 한 스트림에서 난수를 몇 개 뽑든 다른 스트림에 영향이 없다.
"""

from __future__ import annotations

import numpy as np

STREAM_IDS: dict[str, int] = {
    "data": 101,
    "novel_data": 102,
    "init": 201,
    "order": 301,
    "teacher_permutation": 401,
    "diagnostics": 501,
}


def stream(seed: int, name: str) -> np.random.Generator:
    """고정 정수 매핑을 쓰는 독립 난수 발생기를 만든다 (부작용 없음)."""
    if name not in STREAM_IDS:
        raise KeyError(f"등록되지 않은 난수 스트림 이름: {name!r}")
    ss = np.random.SeedSequence([int(seed), STREAM_IDS[name]])
    return np.random.default_rng(ss)


def all_streams(seed: int) -> dict[str, np.random.Generator]:
    return {name: stream(seed, name) for name in STREAM_IDS}
