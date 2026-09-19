# IMPLEMENTATION_STATUS.md — 구현 여부와 검증 상태 (분리)

> **이번 납품의 과학적 수치 실험 상태는 `not_run` 이다.**
> 보고할 정확도·통과율·성능 수치를 만들지 않았고, 실행하지 않은 결과·그래프를
> 만들어 보고하지도 않았다. 과학적 실행은 사용자가 명령을 내릴 때만 일어난다.
>
> **예외 — 자동 실행 기능의 배관 점검.** 사용자가 요청한 "전체 자동 실행"
> (`run-all`) 은 실행 경로가 실제로 도는지 확인하지 않고는 만들 수 없다.
> 그래서 최소 규모 설정에 한해 실행 경로를 직접 돌려 **배관을 점검했다**.
> 무엇을 돌렸고 무엇을 찾았는지는 아래 0절에 전부 적었다. 이 실행으로 얻은
> 수치는 **기능 점검용이며 실험 결과가 아니다.** 저장소에는 실행 결과물을
> 커밋하지 않았다.

---

## 0. 실제로 실행한 것과 그 결과

### 0.1 실행하지 않고 한 점검

1. `python -m compileall` 로 모든 `.py` 파일의 **구문/컴파일 검사**
2. `cortex.config.load(...)` 로 4개 설정 파일의 **스키마 검증** (시뮬레이션 없음)
3. 단일 파일 판(`cortex_all_in_one.py`)에 대해:
   - 경로 import 검사 (작업 디렉터리에 생기는 파일 없음, matplotlib 미로드 확인)
   - `python cortex_all_in_one.py selftest` — 내장 설정 4종이 `configs/*.json` 과
     같은 해석 결과(sha256)를 내는지 대조 (4/4 일치, 시뮬레이션 없음)
   - `--help` 출력 확인 (argparse 만 동작)
   - 패키지와 단일 파일의 **최상위 심볼 집합 대조** (누락 0)
   - `tools/build_single_file.py` 재실행이 바이트 단위로 같은 파일을 만드는지 확인

### 0.2 실제로 돌린 것 (배관 점검)

| 무엇 | 어떻게 | 결과 |
|---|---|---|
| `run-all` 전 단계 | `minimal` 설정, 전체 7단계 | 7/7 `completed` |
| `run-all` 전 단계 | `v1_small`, `hierarchy_small` (자극 수 제한) | 7/7 `completed`, 단 아래 미해결 항목 |
| `run-all --dry-run` | `minimal` | 아무 것도 실행하지 않고 계획만 출력 |
| `run-all --stages` | 일부 단계만 | 폴더 번호가 전체 실행과 같음 |
| 필수 검증 1~14 | `minimal` 설정 | 14/14 통과 (아래 결함 수정 후) |
| pytest | `tests/` 전체 | 76개 통과 |

실행 결과물은 임시 폴더에 쓰고 저장소에 커밋하지 않았다. 패키지 자동 설치와
데이터 자동 다운로드는 **배포 코드에 넣지 않았다** (점검용 pytest 는 작업
환경에만 설치했다).

### 0.3 이 점검으로 찾아서 고친 결함

한 번도 실행된 적이 없던 경로였기에 아래 결함은 모두 배포 전까지 드러나지
않았다. 각 항목에는 `tests/test_autorun_and_drive.py` 의 회귀 테스트가 있다.

| # | 결함 | 증상 | 수정 |
|---|---|---|---|
| 1 | `NeuronRecord` 가 `frozen=True` dataclass 라 property setter 까지 막혔다 | 검증 1 이 `FrozenInstanceError` 로 실패 | 식별 필드는 계속 막고, setter 가 있는 칸만 통과시키는 `__setattr__` 을 클래스 생성 뒤에 건다 (`records.py`) |
| 2 | 검증 5(a) 가 고정 50 스텝만 적분하고 EL 수렴을 요구했다 | 막시간상수에 따라 통과/실패가 갈림 | 후향 오일러 닫힌 해에서 필요한 스텝 수를 계산해 쓰고, 닫힌 해와의 일치도 함께 검사 (`validation.py`) |
| 3 | `local_radius` 가 깊이를 포함한 3D 거리를 써서 층간 규칙이 시냅스를 0개 만들었다 | 검증 9 가 "이름만 존재하는 경로" 로 실패 | 배선 규칙에 `radius_space: cortical_3d\|surface` 를 추가하고 층간 규칙은 표면 접선 거리를 쓰게 했다. 지연은 여전히 3D 거리 (`config.py`, `areas.py`, `configs/*.json`) |
| 4 | 검증 11 의 문맥 전류 300 pA 가 임계를 넘기지 못했다 | "문맥 입력이 활동/학습을 바꾼다" 가 항상 실패 | 3구획 정상상태에서 필요한 전류를 손계산해 쓰고, 약한 전류에서는 막전위 변화를 따로 검사 (`validation.py`) |
| 5 | `encoder.normalize()` 가 `scale[:, None, None]` 로 방송해 `(C,S)` 입력을 `(C,C,S)` 로 키웠다 | `simulate` 가 einsum/인덱싱 오류로 죽거나 조용히 망가짐 | 첫 축이 채널 축이면 차원 수와 무관하게 동작하고, 채널 수가 맞지 않으면 오류 (`retina.py`) |
| 6 | `fit_normalization` 이 `np.percentile` 을 NaN 이 섞인 배열에 썼다 | 0 화소가 하나라도 있으면 **모든** 채널 스케일이 1.0 으로 무너져 정규화가 무효화 | `np.nanpercentile` 로 교체하고 전부 NaN 인 채널을 따로 처리 (`retina.py`) |
| 7 | 스케일을 영상 해상도에서 추정하고 격자 샘플에 적용했다 | 불균일 샘플링의 저역통과 때문에 구동이 수십 배 약해짐 | `fit_normalization(sampler=...)` 로 **적용할 표현에서** 추정하고, 어디서 추정했는지 manifest 에 기록 (`retina.py`, `runner.py`, `validation.py`, `autorun.py`) |
| 8 | 신호가 없는 채널(회색조 자극의 색 대립 채널)이 자기 백분위로 정규화되어 최대 세기까지 증폭됐다 | 내용 없는 채널이 가장 센 채널과 같은 세기로 망막을 구동 | 채널 스케일에 `min_scale_ratio * max(scale)` 바닥을 둔다 (`retina.py`) |
| 9 | `minimal` 의 `sum_mode_scale=1.0` 이 망막 임계(0.35)의 1/7 밖에 못 만들었다 | 오류 없이 **전 회로 스파이크 0건** | 한 구간 최대 기여가 임계를 넘도록 40.0 으로 바꾸고 근거를 `meta.notes` 에 적었다 (`configs/minimal.json`) |

### 0.4 점검으로 드러났지만 **고치지 않은** 것

`v1_small` 과 `hierarchy_small` (전도도 LIF 설정) 은 망막까지는 발화하지만
**LGN 부터 그 뒤가 전부 침묵한다.** 원인은 시냅스 가중치가 표적을 임계까지
올리기에 모자란 것이고, `Retina->LGN` 기준으로 앞 영역이 약 212 Hz 로 발화해야
하는데 망막 구동 상한은 약 60 Hz 다 (약 3.5배 부족).

이것은 코드 결함이 아니라 **모형 파라미터 선택**이므로 임의로 바꾸지 않았다.
대신 조용히 실패하지 않도록 두 가지 진단을 넣었다.

* **실행 전** `manifest.json` 의 `transmission_headroom` — 흥분성 배선 규칙마다
  필요한 앞 영역 발화율과 실제 상한을 손계산해 비교하고, 닿지 못하는 규칙을
  `blocked_rules` 에 적는다. 망막 구동이 자기 임계에 닿지 못하면
  `inspect-config` 단계에서 이미 경고한다.
* **실행 후** `manifest.json` 의 `silent_areas` 와 `SUMMARY_ko.md` 의 경고 —
  모든 표본에서 한 번도 발화하지 않은 영역을 이름으로 적고, "이 0 은 모형의
  결론이 아니다" 라고 명시한다.

가중치를 올리려면 `wiring.rules[].weight.median` 을 키우면 된다. 어느 규칙을
얼마나 올려야 하는지는 `transmission_headroom` 의
`presyn_rate_needed_hz / presyn_rate_max_hz` 비율이 그대로 알려준다.

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
| 필수 검증 1~14 | ✅ 완료 | `cortex/validation.py` | 실제 계산값 사용, 항상 True 검사 없음. `minimal` 에서 14/14 통과 (0절) |
| pytest 테스트 | ✅ 완료 | `tests/` | 76개 통과 (0절) |
| `run.py` 한국어 메뉴 | ✅ 완료 | `run.py` | `A)` 전체 자동 실행 포함 |
| CLI (dry-run 기본) | ✅ 완료 | `cortex/cli.py` | `run-all` 만 기본 실행 |
| **전체 자동 실행 (`run-all`)** | ✅ 완료 | `cortex/autorun.py` | 7단계, 지정 폴더에 결과 작성. README_KO.md 3-1절 |
| 자동 실행 한국어 요약 | ✅ 완료 | `cortex/autorun.py` | `SUMMARY_ko.md` 는 `summary.json` 에서만 생성 |
| 완료된 결과 폴더 보호 | ✅ 완료 | `cortex/autorun.py` | `--overwrite` 없이는 덮어쓰지 않는다 |
| 백엔드 자동 선택 | ✅ 완료 | `cortex/autorun.py` | h5py 없으면 npz 로 바꾸고 **알린다** |
| 자극 수 제한 (`max_stimuli`) | ✅ 완료 | `config.py`, `runner.py` | 자른 사실을 manifest 와 요약에 기록 |
| 구동 여유 진단 (실행 전) | ✅ 완료 | `config.drive_headroom_warnings` | 망막이 자기 임계에 닿는지 손계산 |
| 전달 여유 진단 (실행 전) | ✅ 완료 | `runner.transmission_headroom` | 배선 규칙별 필요 발화율 vs 상한 |
| 침묵 영역 보고 (실행 후) | ✅ 완료 | `runner.silent_area_report` | 스파이크 0 을 결론으로 오해하지 않게 한다 |
| 배선 거리 공간 선택 | ✅ 완료 | `config.py`, `areas.py` | `radius_space: cortical_3d\|surface` |
| 단일 파일 판 (설정 내장) | ✅ 완료 | `cortex_all_in_one.py` | 패키지와 같은 코드. 차이는 README_KO.md 2-1절 |
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

0절의 배관 점검에서 **`minimal` 설정에 한해** 검증 1~14 를 실제로 돌렸다
(14/14 통과). 아래 "실행" 칸은 그 한 설정에 대한 결과이며, 다른 설정에서의
결과는 사용자가 직접 돌려야 한다. `v1_small` / `hierarchy_small` 에서도 14/14 가
통과하지만, 0.4절의 침묵 문제는 검증 항목이 아니라 모형 파라미터 문제다.

| 검사 | 대상 | 구현 | 실행 (`minimal`) |
|---|---|---|---|
| 1 | 3×3 뷰·타입 배열·연결/로그 조회 일치, ID 유효성 | ✅ | 통과 |
| 2 | 중복 반영 방지, 과거 로그 보존, 전도도 잔류 vs 새 입력 | ✅ | 통과 |
| 3 | 예약/도착 시각, 지연 양자화, 동시 사건 순서 불변성 | ✅ | 통과 |
| 4 | threshold 모드 손계산 합과 경계 발화 | ✅ | 통과 |
| 5 | LIF 누설·단일 펄스·불응기·구획 결합·dt 수렴 | ✅ | 통과 |
| 6 | Dale 제약, 전도도 비음수, 학습/클리핑 후 부호·범위 | ✅ | 통과 |
| 7 | 색 변환, DoG 상수 응답, log-polar 왕복·범위·aliasing | ✅ | 통과 |
| 8 | 회전·위치·위상·양안 입력 변화에 따른 반응 변화 | ✅ | 통과 |
| 9 | 층 깊이·세포 유형·필수 배선·연결 수 집계 | ✅ | 통과 |
| 10 | 측정 전후 가중치·상태·흔적·로그 위치·RNG 동일성 | ✅ | 통과 |
| 11 | 자유/유도 0교정 비교, 교사 항/감쇠/항상성 분리 | ✅ | 통과 |
| 12 | Rao 목적함수 미분 vs 유한차분 | ✅ | 통과 |
| 13 | 체크포인트 재개 == 연속 실행, 보고서-기록 일치 | ✅ | 통과 |
| 14 | import 부작용 없음, test 접근 카운터, 분할 중복 없음 | ✅ | 통과 |

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

전 과정을 한 번에 돌리고 결과를 원하는 폴더에 모으려면:

```powershell
python -m cortex.cli run-all --config minimal --out D:/결과폴더
```

`validate` 의 구조적 필수 검사가 실패하면 의존 실험은 중지되고 실패 상태가
기록된다 (`validation.stop_experiment_on_failure`).

결과가 비어 있으면 먼저 `manifest.json` 의 `transmission_headroom` 과
`silent_areas` 를 보라 (0.4절). 스파이크 0 은 대개 모형의 결론이 아니라
구동/전달이 임계에 닿지 못했다는 뜻이다.
