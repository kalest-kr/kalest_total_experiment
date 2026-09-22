"""cortex -- 3x3 뉴런 기록 구조 기반 시각피질 시뮬레이터 패키지.

**import 부작용 없음**: 이 패키지를 import 해도 데이터 생성, 학습, 파일 쓰기,
GPU 초기화, 네트워크 접근이 일어나지 않는다. 하위 모듈도 여기서 import 하지
않으므로 ``import cortex`` 는 상수 정의만 수행한다.

실행 진입점은 두 가지다.

* 프로젝트 루트의 ``run.py`` (한국어 터미널 메뉴)
* ``python -m cortex.cli <subcommand>`` (기본 dry-run, ``--execute`` 시 실행)

이번 납품의 모든 수치 실험 상태는 ``not_run`` 이다. IMPLEMENTATION_STATUS.md 참조.
"""

__version__ = "0.1.0"

#: 이 저장소가 제공하는 실험 상태. 실제 실행 결과는 포함되어 있지 않다.
DELIVERY_EXPERIMENT_STATUS = "not_run"

__all__ = ["__version__", "DELIVERY_EXPERIMENT_STATUS"]
