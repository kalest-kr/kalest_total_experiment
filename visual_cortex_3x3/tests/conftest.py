"""pytest 공통 픽스처.

**이 작업 중에는 테스트를 실행하지 않는다.** 사용자가 직접 실행한다::

    python -m pip install -r requirements-dev.txt
    python -m pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def minimal_cfg():
    from cortex.config import load
    return load(PROJECT_ROOT / "configs" / "minimal.json")


@pytest.fixture(scope="session")
def v1_cfg():
    from cortex.config import load
    return load(PROJECT_ROOT / "configs" / "v1_small.json")


@pytest.fixture()
def minimal_model(minimal_cfg):
    from cortex import rng as rng_mod
    from cortex.runner import build_model
    return build_model(minimal_cfg, rng_mod.from_config(minimal_cfg))
