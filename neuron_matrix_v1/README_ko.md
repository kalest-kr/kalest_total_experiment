# neuron_matrix_v1 — 뉴런 3×3 상태 행렬 · 작은 V1 회로 · 국소 교사 학습

이 프로젝트는 **이산 시점의 가중합·임계 발화 모델**로 작은 V1 회로를 만들고,
뉴런별 목표를 주는 외부 교사와 국소 갱신 규칙으로 L2 입력 연결을 학습시킨 뒤,
**교사를 제거한 상태의 성능**을 대조 실험으로 평가한다.

## 이 구현이 아닌 것 (먼저 읽을 것)

- 뇌 전체를 재현하지 않는다. 실제 전도도, 막전위의 시간 적분, 칼슘 농도,
  생화학적 STDP 를 구현하지 않았다. 입력이 없는 다음 tick 에는 `E=I=0` 으로
  다시 계산하는 **비적분 모델**이다.
- 전체 과제 손실을 연쇄 미분하는 **역전파를 사용하지 않는다**. 대신 뉴런별 목표를
  외부 교사가 제공한다는 **정보상의 가정**을 둔다. 최종 분류 라벨 하나로 전체 계층을
  학습시키는 문제를 푼 것이 아니다.
- L2 목표는 **사람이 설계한 위치·방향 지도**이며 실제 V1 의 측정된 정상 반응이 아니다.
  이 목표를 학습한 것은 사전에 지정한 표현을 학습한 것이고, 라벨 없이 의미를 자동
  추출한 것이 아니다.
- 1000×1000 입력을 같은 전처리 API 로 **처리**하지만, 이는 100만 뉴런 학습 완료를
  뜻하지 않는다. 학습은 64×64 입력 / 뉴런 1024개 / 연결 17408개 회로에서만 했다.
- V2·V4·IT·L5·L6 는 **구현하지 않았다**. 입출력 프로토콜과 설계 설명만 제공하며,
  `src/protocols.py` 의 자리표시자는 호출하면 `NotImplementedError` 를 낸다
  (이름만 붙인 항등 함수를 두지 않았다).
- 억제 중계는 "L4 발화를 한 tick 뒤에 전달"하는 단순화이며 실제 억제성 세포의
  다양성을 재현한 것이 아니다.
- 이 회로의 수치를 과거 DTP/KP 등 다른 구조의 실험 수치와 직접 비교하지 않는다.

## 설치와 실행

```bash
python3 -m pip install -r requirements.txt     # numpy, scipy, matplotlib, pillow

python run_checks.py                            # 필수 검증 T1~T10
python -m unittest discover -s tests            # 부가 단위 검사 37개

python run_experiment.py --profile quick        # seed 42, 5 에폭 (형식 확인용)
python run_experiment.py --profile main         # seed 42~46, 25 에폭 (본 실험)
python run_experiment.py --profile quick --lr-search   # 선택: 학습률 탐색

python make_report.py --profile main            # 한국어 보고서 자동 생성

python run_demo.py --model results/main/models/local_learning_seed42
python run_demo.py --model results/main/models/local_learning_seed42 --headless
python run_demo.py --model results/main/models/local_learning_seed42 --headless --show-targets
```

Windows / PyCharm 에서도 같은 명령으로 실행된다 (경로 구분자와 GUI 유무만 다르다).
GUI 가 없는 환경에서는 `run_demo.py` 가 자동으로 headless 로 전환해 PNG 를 저장한다.
PyTorch / GPU 는 **필수 의존성이 아니며 이 프로젝트는 사용하지 않는다.**

## 뉴런의 3×3 행렬

```
M_i = [[x_i, y_i,         z_i],
       [u_i, threshold_i, P_i],
       [E_i, I_i,         b_i]]
```

| 셀 | 의미 | 단위 |
|---|---|---|
| (0,0) (0,1) (0,2) | x, y, z — 모형 피질 좌표 | 모형 좌표 (설계값, 조직 측정 아님) |
| (1,0) | u — 현재 시점의 발화 판단 입력 | 무차원 |
| (1,1) | threshold — 발화 임계값 | 무차원 |
| (1,2) | P — 송신 신호에 곱하는 전달 계수 | 무차원 |
| (2,0) | E — 가중치 적용 흥분 합 + `external_sensory_drive` | 무차원 |
| (2,1) | I — 가중치 적용 억제 입력 **크기** 합 (양수 저장) | 무차원 |
| (2,2) | b — `b_base + correction_input` | 무차원 |

- 행렬은 `float64`, shape `(3,3)` 이다. 안에 리스트·사전·다른 뉴런 객체를 넣지 않는다.
- 서로 다른 의미·단위의 항목을 한 번에 정규화하거나 행렬 전체에 학습용 행렬곱을 적용하지 않는다.
- `Neuron.threshold`, `Neuron.excitatory_sum` 등은 **해당 셀을 읽고 쓰는 property** 이며
  같은 값을 별도 필드에 중복 저장하지 않는다.
- 행렬 밖 메타데이터: `neuron_id, name, area, layer, cell_type,
  preferred_orientation_deg, receptive_field, position_space, fired,
  observation_tick, incoming_connection_ids, outgoing_connection_ids`.
- **피질 좌표 (x,y,z) 와 원본 영상의 수용장 좌표는 완전히 다른 좌표계다.**

### 구현 선택: 행렬 뷰 + 벡터 배열

모든 뉴런의 3×3 행렬은 `NeuronStateStore` 가 소유한 하나의 `(N,3,3) float64`
배열 위의 **뷰**다. 따라서 명세대로 뉴런마다 `(3,3) float64` 행렬을 가지면서도
엔진이 전체 모집단을 NumPy 로 벡터화해 계산할 수 있다. `fired`, `b_base`,
`correction_input`, `external_sensory_drive`, `observation_tick` 도 각각 배열
1개씩에만 저장하고 `Neuron` 은 property 로 그 셀에 접근한다 (중복 저장 없음).

## 계산 정의 (명세 [4])

```
E_i = external_sensory_drive_i + Σ_k(excitatory) weight_k * s_k
I_i = Σ_k(inhibitory) weight_k * s_k
b_i = b_base_i + correction_input_i
u_i = E_i - I_i + b_i
q_i = 1 if u_i >= threshold_i else 0
emitted_value_i = P_i * q_i
```

- 수신 측에서는 `emitted_value` 에 연결 가중치만 적용한다. 송신 P 를 다시 곱하지 않는다.
- 교사 교정은 `external_sensory_drive` 를 바꾸지 않고 `b` 의 `correction_input` 으로만 들어간다.
- `q=0` 이면 전송 이벤트를 생략하며, 생략된 값은 다음 계산에서 0 이다.

기준 예제 (`run_checks.py` T2 에서 재현): A→D exc 0.6×1.0, B→D exc 0.4×0.5,
C→D inh 0.6×1.0, b=0, threshold=0.5 → **E=0.8, I=0.6, u=0.2, q=0**.
이 결과를 근거로 C 가 잘못된 연결이라고 단정하지 않는다.

## 동기식 실행 순서 (`SimulationEngine.step`)

1. 현재 tick 의 도착 이벤트와 외부 자극을 수집
2. 연결별 도착값 `s_k` 정리
3. 모든 뉴런의 E, I, b, u 계산
4. **같은 가중치 상태**에서 모든 뉴런의 q 를 일괄 결정
5. 발화 뉴런의 출력을 `delay_ticks` 에 따라 미래 큐에 삽입
6. 관찰 시점 스냅샷 저장
7. tick 증가

한 tick 안에서 연쇄 발화는 없다. E/I 누적은 `(post_name, kind, pre_name, connection_id)`
정준 순열로 수행하므로 **뉴런 등록 순서를 바꿔도 부동소수점 결과까지 동일**하다.

`reset_transient_state()` 는 버퍼·이벤트 큐·E/I/u·발화·교정을 초기화하고
**학습 가중치와 고정 메타데이터는 유지**한다. `reset_parameters()` 는 threshold/P/
위치/b_base/가중치를 구성 직후 값으로 되돌린다.

## 생물학적 층 이름과 이 구현의 계산 역할

| 이름 | 이 구현에서 실제로 하는 일 | 관찰 tick | 학습 | 주의 |
|---|---|---|---|---|
| **L4** (ON/OFF) | 격자점마다 ON/OFF 1개. 정규화된 전처리 값을 tick 0 의 `ExternalDrive` 로 받아 threshold 0.1 로 이진 발화 | 0 | 없음 | 이진화로 소실된 정보를 뒤의 교사가 복구해 준다고 가정하지 않는다 |
| **INH 중계** | L4 마다 1개. L4→INH 는 weight=1 고정 흥분(지연 1). P=1, threshold=0.5. INH→L2 는 억제 연결(지연 1) | 1 | INH→L2 가중치만 학습 | 실제 억제성 세포의 다양성을 재현한 것이 아니라 "한 tick 뒤 전달" 단순화 |
| **L2** (0°/90° 후보) | 원본 좌표에서 가까운 L4 격자점 k≤16 개의 ON/OFF 를 직접 흥분(지연 2)으로, 대응 INH 를 억제(지연 1)로 받아 tick 2 에 판정 | 2 | **학습 대상 (이 회로의 유일한 학습 연결)** | 같은 위치의 두 방향 후보는 같은 입력 집합에서 시작하고 가중치만 다르다. 정답 방향 필터를 미리 복사하지 않는다 |
| **L3** | 같은 방향의 가까운 L2 3개를 각 1/3 고정 가중치로 모으고 threshold 0.5 | 3 | 없음 (고정) | 신호 전달·관찰 확인용 보조 출력. 위상 불변성이나 도형 인식을 주장하지 않는다 |
| **L1** | 교정 신호가 들어오는 **논리적 인터페이스** (`L1Relay`) | — | — | 새 피라미드 세포체 집단을 만들어 교사 역할을 주지 않았다 |
| L5 / L6, V2·V4·IT | **미구현** | — | — | 입출력 프로토콜과 설계 설명만 (`src/protocols.py`) |

지연 정렬(L4→L2 직접 2 tick, L4→INH 1 tick, INH→L2 1 tick)은 두 경로가 L2 의
tick 2 에 함께 도착하도록 맞춘 **설계값**이다.

## 자료구조

| 구조 | 파일 | 역할 |
|---|---|---|
| `ConnectionTable` | `src/connections.py` | 지속되는 연결 목록 (`connection_id, pre_id, post_id, kind, weight, delay_ticks, trainable`). weight 는 비음수 크기이고 부호는 kind 가 결정한다. 같은 pre/post 사이 복수 연결 허용 |
| `InputEvent` / `InputBuffer` | `src/events.py` | 일시적 도착 이벤트. 수신 함수는 **저장만** 하고 E/I 누적이나 발화를 하지 않는다. 처리한 버퍼는 비운다 |
| `ExternalDrive` | `src/events.py` | 전처리기의 연속값 주입. 신경 연결로 위장하지 않는다 |
| `TraceRecorder` | `src/diagnostics.py` | 선택 뉴런·선택 시행·최대 행 수로 제한된 진단 기록. 학습 함수는 이 기록을 읽지 않는다 |

## 국소 학습 규칙 (명세 [9])

```
d_i        = target_i - q_i_free
e_k        = s_k_free
sign_k     = +1 (excitatory) / -1 (inhibitory)
Z_i        = epsilon + Σ_k(trainable inputs to i) e_k²
G_k        = learning_rate * d_i * sign_k * e_k / Z_i
weight_new = clip(weight_old + G_k, 0, weight_max)
```

기본값 `epsilon=1e-8, learning_rate=0.05, weight_max=1.0`.

- 이것은 이 자료구조를 실험하기 위한 **제한된 퍼셉트론형 기준 규칙**이다.
  생화학적 학습 법칙도, 손실 하강이 보장된 심층 알고리즘도 아니다.
- postsynaptic q 를 다시 곱하지 않으므로 발화하지 못한 뉴런도 학습한다.
- `e` 는 STDP 나 장기 누적 흔적이 아니다. 이전 영상 값을 누적하지 않는다.
- `d` 는 이진 활동 차이다. 로짓 오차나 손실 기울기가 아니다.
- 모든 G 는 같은 자유 실행 스냅샷에서 계산해 한꺼번에 적용한다. 한 영상의 순방향
  계산 도중에는 가중치를 바꾸지 않는다. 학습 단위는 영상 1장이다.
- threshold / P / b_base 는 학습하지 않고 가중치 감쇠도 쓰지 않는다.
- 도착 입력이 전부 0 이면 갱신은 0 이다. 이를 숨기려고 임계값이나 기저 입력을
  자동 조정하지 않는다.
- `Z` 는 해당 수신 뉴런의 지역 입력만 사용한다.
- `compute_local_update_for_neuron(d_i, e, sign, trainable, cfg)` 는 **순수 국소 함수**이며
  전체 라벨·다른 뉴런의 오차·감사 로그에 접근하지 않는다. 벡터 구현이 이 함수와
  같은 값을 내는지는 T5 에서 검사한다.

### 활동의 일시적 교정과 학습의 분리 (명세 [10])

```
if target_i == 1: c_i = max(0, threshold_i + margin - u_i_free)
else:             c_i = min(0, threshold_i - margin - u_i_free)
```

자유 실행과 유도 실행은 같은 tick 수·연결·지연·임계값·P 를 쓰고 두 실행 사이에
가중치를 바꾸지 않는다. 학습 규칙의 `q` 와 `e` 는 **자유 실행**에서 가져온다.
**유도 중 정확도는 최종 성능으로 보고하지 않는다.**
`error_override_zeros=True` 이면 학습용 `d` 와 교정 `c` 가 모두 0 이 되고,
가중치가 비트 단위로 보존되는지 T5 에서 검사한다.

## 전처리 (명세 [6])

1. 입력을 [0,1] 실수로 변환 (uint8 은 /255, float 는 범위 검사)
2. RGB 는 sRGB 선형화 후 Rec.709 휘도 근사로 회색조.
   float 입력이 선형값인지 sRGB 인지는 `input_colorspace` 인자로 **명시**한다.
   → 이것은 휘도 한 경로일 뿐 완전한 망막 색채 회로가 아니다.
3. 정규화된 가우시안 두 개의 차 (`sigma_center=1`, `sigma_surround=3`,
   `mode='reflect'`, `truncate=3`)
4. `ON=max(DoG,0)`, `OFF=max(-DoG,0)`
5. 로그-극좌표 격자(기본 `radial_bins=8`, `angular_bins=16`, `r0=2px`)에서
   `sigma_pool = max(0.5, 0.5*max(Δr, r_k*2π/angular_bins))` 으로 저역통과 후
   이중선형 보간 표본 추출 (근사 없는 `direct` 경로)

기본 시야는 영상에 **내접한 원**이다. 바깥 모서리는 포함되지 않으며 마스크와 그림에
표시한다. 모형 수용장 반경은 `max(2, 2*sigma_pool)` 픽셀이며 이는 **실험 설계값**이다.

L4 입력 정규화 계수는 **훈련 데이터로만** 추정한다 (양수 ON/OFF 값의 95 백분위로
나눈 뒤 [0,1] 제한). 모든 값이 0 이면 `scale=1.0`, `degenerate=True` 로 기록한다.
검증·시험 영상으로 계수를 다시 추정하지 않는다.

로그-극좌표 배열 인덱스와 원본 영상의 선 기울기를 혼동하지 않는다. 방향 라벨과
목표 생성은 **원본 영상 좌표계** 기준이다.

## 교사 (명세 [8])

학생은 이미지 픽셀만 받는다. 도형의 각도·좌표·정답 활동 지도는 교사와 평가기만 읽는다.

- 빈 화면 → `target_i = 0`
- `d_i` = 수용장 중심과 **유한 선분**의 최소 거리
- 방향 차이는 180° 주기의 최소 차이
- `d_i ≤ rf_radius_i/2` 이고 방향차 ≤ 15° 이면 `target_i = 1`

이 교사가 **이미 알고 있는 정답 정보**: 선분 좌표와 방향, 각 L2 의 수용장과 선호
방향, 그것들로 만든 뉴런별 이진 목표, 그리고 교정값을 어느 뉴런에 보낼지.
IT 의 최종 오차 하나에서 이 지도를 자동 역산한 기능은 **없다**.

`teacher_feasibility_report` 는 같은 L4 발화 패턴에 서로 다른 목표가 붙은 표본을
세어 **식별 불가능성 후보**로 기록한다. 모든 교사 목표가 학생에게 실현 가능하다고
가정하지 않는다.

## 대조 실험 (명세 [13])

동일 데이터·연결·초기 가중치·표본 순서에서 조건만 바꾼다 (T8 에서 검사).

| 조건 | 내용 | 최종 평가 |
|---|---|---|
| `frozen` | 가중치 고정, 교정 없는 추론 | 교정 없음 |
| `correction_only` | 훈련 자극에서 활동 교정만 시연, 가중치 고정 | 교정 없음 |
| `local_learning` | 지역 교사 + 국소 규칙으로 연결 학습 | 교정 없음 |
| `shuffled_teacher` | 훈련 표본 간 목표 지도를 고정 무작위 순열로 교체 (검증·시험 목표는 섞지 않음) | 교정 없음 |

주 지표는 **balanced accuracy 와 F1** 이다. 불활성 뉴런이 많으므로 전체 비트
정확도만으로 성공을 주장하지 않는다. 항상 비활성 / 항상 발화 기준값도 함께 보고한다.

난수 스트림은 `data, novel_data, init, order, teacher_permutation, diagnostics` 로
분리되어 있고 각각 고정 정수에 매핑된 `SeedSequence([seed, id])` 에서 만들어진다
(Python 의 실행별 hash 값에 의존하지 않는다).

## 프로젝트 구조

```
neuron_matrix_v1/
  README_ko.md          이 문서
  requirements.txt
  config.json           모든 기본값과 채택한 구현 선택 기록
  src/
    neuron.py           3x3 행렬 뉴런, 상태 저장소, 수용장
    connections.py      ConnectionTable (검증 포함)
    events.py           InputEvent, ExternalDrive, 수신 버퍼
    engine.py           SimulationEngine.step / run_trial / reset_*
    retina.py           회색조 + DoG + ON/OFF
    spatial_mapping.py  로그-극좌표 격자, 저역통과 표본 추출, 입력 정규화
    v1.py               V1Circuit.build
    teacher.py          LocalTeacher, L1Relay, 실현 가능성 진단
    local_learning.py   국소 규칙 (순수 함수 + 배치 구현)
    diagnostics.py      TraceRecorder, inspect_neuron, counterfactual_check
    datasets.py         합성 장면 생성과 렌더링
    metrics.py          balanced accuracy / F1 / 분해 지표
    persistence.py      모델 저장·복원, 해시
    experiment.py       조건별 실험 실행기
    visualization.py    그림
    protocols.py        V2/V4/IT 입출력 프로토콜 (미구현 자리표시자)
  run_checks.py         필수 검증 T1~T10
  run_experiment.py     대조 실험
  run_demo.py           저장 모델 추론 데모
  make_report.py        한국어 보고서 자동 생성
  tests/test_units.py   부가 단위 검사
  results/              실행 결과 (JSON/CSV/PNG/체크포인트/보고서)
```

## 공개 인터페이스

```python
Neuron, Population, NeuronStateStore, ReceptiveField
ConnectionTable
InputEvent, ExternalDrive, InputBuffer, ExternalDriveBuffer
SimulationEngine.step() / .run_trial() / .reset_transient_state() / .reset_parameters()
RetinaEncoder.encode(image, input_colorspace)
LogPolarSampler.sample(channels)
InputNormalizer.fit / .transform
V1Circuit.build(config, seed, init_rng)
LocalTeacher.make_targets(scene_metadata, neurons) / .make_corrections / .make_errors
L1Relay.route(corrections)
LocalLearner.compute_updates(free_snapshot, local_errors) / .apply_updates(updates)
compute_local_update_for_neuron(d_i, e, sign, trainable, cfg)   # 순수 국소 함수
TraceRecorder.inspect_neuron(trial_id, neuron_id, tick)
counterfactual_check(engine, drive, interventions, observe_ids, ...)
connection_role_table(...)
save_model / load_model
```

각 함수의 인자, 반환 shape, 단위, 부작용 여부는 docstring 에 적혀 있다.
`run_trial` 과 분석 함수는 가중치를 바꾸지 않는다. 학습 함수와 결과 평가기는 분리되어 있다.

## 필수 검증

| 검사 | 내용 |
|---|---|
| T1 | 행렬 shape/dtype, 셀 의미와 속성 일치, 잘못된 연결 거부 |
| T2 | A/B/C→D 예제 재현, 저장 기여 합 == 저장 E/I/u |
| T3 | P 가 정확히 한 번 적용, 복수 연결·복수 이벤트, 지연 도착 tick, 마지막 tick q 오독 방지 |
| T4 | 등록 순서 불변, 재실행 불변, 영상 간 상태 누출 없음, 초기화가 가중치를 지우지 않음, bulk/object 경로 동일 |
| T5 | 학습 부호 방향, d=0 과 0교정에서 비트 단위 보존, 입력 0 연결 G=0, clipping 이 종류를 안 뒤집음, 벡터==순수 함수 |
| T6 | 두 입력 [1,0]→1 / [0,1]→0 의 작은 학습 예제, 교사 제거 후 유지 |
| T7 | 진단 on/off 학습 결과 동일, 반사실 개입 후 상태·RNG 보존 |
| T8 | 라벨만 바꿔도 감각 경로 불변, 분할 간 scene_id 비중복, 조건 간 초기값·순서 동일 |
| T9 | 저장·복원 후 구조·계수·교사 없는 예측 일치 |
| T10 | 64×64 및 1000×1000 처리, 좌표 범위·유효 시야·shape·유한값, 소요 시간 기록 |

임계 함수에 일반 유한차분으로 미분 가능성을 요구하거나 모든 학습 시행의 손실 감소를
필수 통과 조건으로 요구하지 않는다. 구현 검사가 실패하면 원인을 고치고, 학습 가설이
실패하면 결과를 그대로 남긴다.

## 산출물

`results/<profile>/` 아래에 `run_summary.json`(집계·환경·시간·해시),
`raw/*.json`·`raw/*_predictions.npz`(조건·시드별 원자료와 **동일 예측 배열**),
`metrics.csv`, `models/*`(체크포인트), `figures/*.png`,
`diagnostics/*`(진단 기록), `REPORT_ko.md`(자동 생성 한국어 보고서)가 생성된다.
`results/checks.json` 은 필수 검증 결과다.
