# IMPLEMENTATION_STATUS.md — 구현 여부와 검증 상태 (분리)

> **이번 납품의 모든 수치 실행 상태는 `not_run` 이다.**
> 학습, 시뮬레이션, 파일럿, 성능 탐색, 데이터 생성, 수치 검증, 그래프 생성은
> 이 저장소를 만드는 동안 **한 번도 실행하지 않았다.** 사용자가 명령을 실행할 때만
> 수행된다. 실행하지 않은 결과·정확도·통과율·그래프를 만들어 보고하지 않았다.

이 저장소를 작성하는 동안 실제로 한 것은 다음 두 가지뿐이다.

1. `python -m compileall` 로 모든 `.py` 파일의 **구문/컴파일 검사**
2. `cortex.config.load(...)` 로 4개 설정 파일의 **스키마 검증** (시뮬레이션 없음)
3. 단일 파일 판(`cortex_all_in_one.py`)에 대해:
   - 경로 import 검사 (작업 디렉터리에 생기는 파일 없음, matplotlib 미로드 확인)
   - `python cortex_all_in_one.py selftest` — 내장 설정 4종이 `configs/*.json` 과
     같은 해석 결과(sha256)를 내는지 대조 (4/4 일치, 시뮬레이션 없음)
   - `--help` 출력 확인 (argparse 만 동작)
   - 패키지와 단일 파일의 **최상위 심볼 집합 대조** (228개 전부 존재, 누락 0)
   - `tools/build_single_file.py` 재실행이 바이트 단위로 같은 파일을 만드는지 확인

패키지 자동 설치와 데이터 자동 다운로드도 하지 않았다.

---

## 1. 구현 상태

| 항목 | 구현 | 위치 | 비고 |
|---|---|---|---|
| 3×3 기록 인터페이스 (`as_matrix`) | ✅ 완료 | `cortex/records.py` | object 배열, 미러 없음 |
| 타입 배열(SoA) 상태 저장소 | ✅ 완료 | `cortex/records.py` | |
| 희소 시냅스 edge table + CSR | ✅ 완료 | `cortex/synapses.py` | N×N 밀집 행렬 없음 |
| 입력 이벤트 로그 + 뉴런별 조회 | ✅ 완료 | `cortex/events.py` | full/selected/summary |
| 지연 큐와 양자화 | ✅ 완료 | `cortex/events.py` | 최소 1 스텝 강제 |
| `sum_threshold` 모드 | ✅ 완료 | `cortex/dynamics.py` | 무차원, 순서 불변 |
| `conductance_lif` 모드 (3구획) | ✅ 완료 | `cortex/dynamics.py` | 후향 오일러 닫힌 해 |
| AMPA / GABA_A | ✅ 완료 | `cortex/dynamics.py` | 단일 지수 감쇠 |
| NMDA (Mg 차단) | ⚠️ 부분 | `cortex/dynamics.py` | `B(V)` 를 스텝 시작 전압에서 평가한 **선형화**. 상승 시정수 없음 |
| 불응기·리셋·구획 결합 | ✅ 완료 | `cortex/dynamics.py` | `refractory_input_policy` 지원 |
| 이온 채널 / 칼슘 / 수상돌기 스파이크 | ❌ 미구현 | — | 구현했다고 표시하지 않는다 |
| 단기 가소성 / 방출 확률 | ❌ 미구현 | — | `P` 는 출력 이득이지 방출 확률이 아니다 |
| sRGB 선형화, RGB→XYZ→LMS | ✅ 완료 | `cortex/retina.py` | 행렬 수치·정규화·출처 기록 |
| 대립 채널 + DoG + ON/OFF (6채널) | ✅ 완료 | `cortex/retina.py` | 커널 합 1 정규화 |
| 저주파 LMS 별도 경로 (3채널) | ✅ 완료 | `cortex/retina.py` | `keep_lowpass_lms` |
| rate / Poisson 입력 | ✅ 완료 | `cortex/retina.py`, `dynamics.py` | 외생 사건열 재생 가능 |
| 로그-극좌표 사상 + 역변환 | ✅ 완료 | `cortex/retinotopy.py` | 중심 Cartesian 패치 |
| 저역통과 후 샘플링 (anti-alias) | ✅ 완료 | `cortex/retinotopy.py` | bin 별 σ, 여유 지표 제공 |
| 균일 샘플링 대조군 | ✅ 완료 | `cortex/retinotopy.py` | 총 샘플 수 맞춤 |
| 3D 배치·층·세포 유형 | ✅ 완료 | `cortex/anatomy.py` | 좌표가 배선·지연에 실제 사용 |
| pinwheel 방향 지도 | ✅ 완료 | `cortex/anatomy.py` | 위치가 선호 방향을 결정 |
| 안구 우세 지도 (별도 속성) | ✅ 완료 | `cortex/anatomy.py` | |
| hypercolumn 미세 좌표 | ✅ 완료 | `cortex/anatomy.py` | 위상 배정에 사용 |
| 배선 규칙 (rf_knn / local_radius / all_to_all) | ✅ 완료 | `cortex/areas.py` | cKDTree 후보 탐색 |
| Gabor 초기 배선 (기록됨) | ✅ 완료 | `cortex/areas.py` | 음의 계수를 ON/OFF 극성으로 처리 |
| 고정 Gabor 대조 경로 + 에너지 | ✅ 완료 | `cortex/v1_reference.py` | "학습" 이라고 하지 않는다 |
| L1 = 교정/피드백 표적 공간 | ✅ 완료 | `anatomy.py`, 설정 배선 | 독립 정답 계산기가 아니다 |
| L6→LGN 피드백 | ✅ 완료 | 설정 배선 | |
| V2/V3/V4/IT 영역 + 재귀 | ✅ 완료 | `configs/hierarchy_small.json` | 실질적 입력·상태·출력을 가진다 |
| V1↔V2↔V4↔IT 기본 경로 + V2↔V3↔V4 | ✅ 완료 | 같은 설정 | 한 사슬 가정 없음, 우회 경로 포함 |
| `learning = none` | ✅ 완료 | `cortex/plasticity.py` | 각 항을 개별로 끌 수 있다 |
| `stdp_homeostasis` | ✅ 완료 | `cortex/plasticity.py` | 흔적 기반, 동시 발화 정책 3종 |
| 임계 적응 (항상성) | ✅ 완료 | `cortex/plasticity.py` | 세포 유형별 목표, 상하한 |
| 이웃 임계 평균화 | ✅ 완료 (기본 꺼짐) | `cortex/plasticity.py` | **소거 실험 옵션 전용** |
| `rao_reference` 연속값 엔진 | ✅ 완료 | `cortex/predictive_coding.py` | 공유 상위 표현, 동기 정착 |
| Rao 유한차분 검사 | ✅ 완료 | `cortex/predictive_coding.py` | 연속값 모델에만 해당 |
| Rao 오차 → apical 전류 결합 | ⚠️ 인터페이스만 (기본 꺼짐) | `predictive_coding.py` | 가정 목록을 반환한다. 좌표 대응은 사용자가 정해야 한다 |
| 분류 readout (ridge) | ✅ 완료 | `cortex/analysis.py` | 라벨/기울기는 여기서만 사용 |
| 로지스틱 readout | ❌ 미구현 | — | 설정 값 `classifier="logistic"` 은 아직 ridge 로 처리되지 않는다 (아래 참조) |
| `ExperimentRunner` / `RunRecorder` / `CheckpointManager` | ✅ 완료 | `runner.py`, `recording.py` | |
| manifest / events / states / updates / metrics / log / errors | ✅ 완료 | `cortex/recording.py` | HDF5 기본, npz 대체 |
| 체크포인트 저장·복원·호환성 검사 | ✅ 완료 | `cortex/recording.py` | 코드/설정 해시 비교 |
| 중단 시 안전 flush | ✅ 완료 | `RunRecorder.__exit__` | KeyboardInterrupt/예외 모두 |
| `report.md` 자동 생성 | ✅ 완료 | `cortex/analysis.py` | 저장된 기록만 읽는다 |
| `explain_neuron` | ✅ 완료 | `cortex/analysis.py` | 관측 기여 분석 |
| 소거(ablation) 재실행 비교 | ✅ 완료 | `cortex/analysis.py` | 인과 주장을 위한 짝지은 비교 |
| 시각화 (배치/지도/타임라인/도달시점/튜닝/Rao) | ✅ 완료 | `cortex/visualization.py` | 별도 명령, 기록만 읽음 |
| 필수 검증 1~14 | ✅ 완료 (미실행) | `cortex/validation.py` | 실제 계산값 사용, 항상 True 검사 없음 |
| pytest 테스트 | ✅ 완료 (미실행) | `tests/` | |
| `run.py` 한국어 메뉴 | ✅ 완료 | `run.py` | |
| CLI (dry-run 기본) | ✅ 완료 | `cortex/cli.py` | |
| 단일 파일 판 (설정 내장) | ✅ 완료 (미실행) | `cortex_all_in_one.py` | 패키지와 같은 코드. 차이는 README_KO.md 2-1절 |
| 단일 파일 생성 도구 | ✅ 완료 | `tools/build_single_file.py` | 재생성 결과가 바이트 단위로 같음을 확인 |
| Numba 가속 | ❌ 미구현 | — | 선택 사항이며 도입하지 않았다. 이름만 나열하지 않는다 |
| GPU / PyTorch | ❌ 미사용 | — | 필수 의존성이 아니다 |

### 알려진 간극

1. `readout.classifier = "logistic"` 은 설정 스키마에 있지만 구현된 것은 ridge
   뿐이다. `train_readout` 은 항상 ridge 를 쓰고 결과에 `"classifier": "ridge"` 를
   기록한다. 로지스틱을 쓰려면 구현을 추가해야 한다.
2. NMDA 의 Mg 차단은 스텝 시작 전압에서 평가한 선형화다. 완전 암시적 해가 아니다.
3. `experiment.protocol` 의 `"sweep"` 값은 스키마에 있으나 별도 분기가 없다
   (`single_pass` 와 같게 동작한다). 스윕은 자극 목록으로 표현한다.
4. VIP 계열 억제 뉴런과 L1 억제 아형 구분은 넣지 않았다 (`l1_inhibitory` 하나).
5. 운동 선택성 전용 시간 회로가 없다. 프레임 시퀀스 자극은 만들 수 있다.

---

## 2. 검증 상태

**모든 항목이 `not_run` 이다.** 아래는 "검사가 존재한다" 는 뜻이지
"통과했다" 는 뜻이 **아니다**.

| 검사 | 대상 | 구현 | 실행 |
|---|---|---|---|
| 1 | 3×3 뷰·타입 배열·연결/로그 조회 일치, ID 유효성 | ✅ | `not_run` |
| 2 | 중복 반영 방지, 과거 로그 보존, 전도도 잔류 vs 새 입력 | ✅ | `not_run` |
| 3 | 예약/도착 시각, 지연 양자화, 동시 사건 순서 불변성 | ✅ | `not_run` |
| 4 | threshold 모드 손계산 합과 경계 발화 | ✅ | `not_run` |
| 5 | LIF 누설·단일 펄스·불응기·구획 결합·dt 수렴 | ✅ | `not_run` |
| 6 | Dale 제약, 전도도 비음수, 학습/클리핑 후 부호·범위 | ✅ | `not_run` |
| 7 | 색 변환, DoG 상수 응답, log-polar 왕복·범위·aliasing | ✅ | `not_run` |
| 8 | 회전·위치·위상·양안 입력 변화에 따른 반응 변화 | ✅ | `not_run` |
| 9 | 층 깊이·세포 유형·필수 배선·연결 수 집계 | ✅ | `not_run` |
| 10 | 측정 전후 가중치·상태·흔적·로그 위치·RNG 동일성 | ✅ | `not_run` |
| 11 | 자유/유도 0교정 비교, 교사 항/감쇠/항상성 분리 | ✅ | `not_run` |
| 12 | Rao 목적함수 미분 vs 유한차분 | ✅ | `not_run` |
| 13 | 체크포인트 재개 == 연속 실행, 보고서-기록 일치 | ✅ | `not_run` |
| 14 | import 부작용 없음, test 접근 카운터, 분할 중복 없음 | ✅ | `not_run` |

`skipped` 는 `passed` 에 포함하지 않는다. 성공률을 맞추려고 기준을 사후에
바꾸지 않는다. 기능/학습 가설에서 기대한 개선이 없다는 이유로 결과를 지우지 않는다.

**검사 13 의 "보고서-기록 일치" 부분**은 실행 기록이 있어야 의미가 있다.
`validate` 만 단독 실행하면 그 부분은 기록 없음으로 처리된다.

---

## 3. 성능에 대해

* 이 저장소는 **측정하지 않은 속도를 "최적" 또는 "빠르다" 고 말하지 않는다.**
* `inspect-config` 는 뉴런 수, 시냅스 수, 이벤트 수, RAM/디스크 사용량을 **추정**
  하지만 `runtime_seconds_estimated` 는 항상 `null` 이다.
* 벡터화(NumPy), 희소 인덱스(scipy.sparse CSR), 공간 인덱스(cKDTree),
  청크 저장(HDF5)을 사용했다. 이것은 설계 선택이지 성능 주장이 아니다.

---

## 4. 사용자가 먼저 실행할 것

```powershell
python -m pip install -r requirements.txt
python run.py
```

또는 CLI 로:

```powershell
python -m cortex.cli inspect-config --config configs/minimal.json
python -m cortex.cli validate --config configs/minimal.json --execute
```

`validate` 의 구조적 필수 검사가 실패하면 의존 실험은 중지되고 실패 상태가
기록된다 (`validation.stop_experiment_on_failure`).
