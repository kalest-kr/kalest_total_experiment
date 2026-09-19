# 뉴런 3×3 행렬 · 작은 V1 회로 · 국소 교사 학습 — 실행 결과 보고서 (quick 프로필)

- 생성 시각(UTC): 2026-09-15T03:16:48Z
- 실행 소요 시간: 20.2 초
- 시드: [42] · 에폭: 5 · 조건: ['frozen', 'correction_only', 'local_learning', 'shuffled_teacher']
- 학습률: 고정 0.05
- **이 보고서는 run_summary.json 에서 자동 생성되었다.** 수치는 실제 실행 결과다.

> 주의: 이 구현은 이산 시점의 가중합·임계 발화 모델이다. 실제 전도도, 막전위의 시간 적분,
> 칼슘 농도, 생화학적 STDP 를 구현한 모델이 아니다. 전체 과제 손실을 연쇄 미분하는 역전파는
> 사용하지 않았고, 대신 **뉴런별 목표를 외부 교사가 제공한다는 정보상의 가정**을 둔다.
> 뇌 전체를 재현했다고 주장하지 않는다.

## 1. 실행 환경과 실행 명령

| 항목 | 값 |
|---|---|
| Python | 3.11.15 (main, Mar  3 2026, 09:26:23) [GCC 13.3.0] |
| 플랫폼 | Linux-6.18.44-fc-v33-x86_64-with-glibc2.39 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.2 |
| Pillow | 12.3.0 |
| PyTorch | 미설치 (필수 의존성 아님) |
| GPU | 미사용 (CPU 전용 실행) |

```bash
python run_checks.py
python run_experiment.py --profile quick
python run_experiment.py --profile main
python make_report.py --profile main
python run_demo.py --model results/main/models/local_learning_seed42 --headless
```

## 2. 구현한 것과 구현하지 않은 것

- 구현된 영역: **V1** (L4 / 억제 중계 / L2 / L3, L1 은 교정 인터페이스)
- 구현하지 않은 영역: **V2, V4, IT, L5, L6** — 입출력 프로토콜과 설계 설명만 제공하며,
  호출하면 `NotImplementedError` 를 낸다. 이름만 붙인 항등 함수를 두지 않았다.
- 구현된 것은 V1 의 L4/INH중계/L2/L3 뿐이다. V2/V4/IT/L5/L6 는 입출력 프로토콜과 설계 설명만 있으며 호출하면 NotImplementedError 를 낸다. 100만 화소 입력 지원은 100만 뉴런 학습 완료를 뜻하지 않는다.

| 회로 규모 | 값 |
|---|---|
| 뉴런 수 | 1024 |
| 연결 수 | 17408 |
| 학습 대상 연결 수 | 16384 |
| 로그-극좌표 격자점 | 128 |
| L2 입력 이웃 수 k | 16 |
| L3 풀링 k | 3 |
| 방향 채널 | [0.0, 90.0] |

## 3. 필수 검증 (T1~T10) 결과

| 검사 | 내용 | 결과 | 세부 단언 | 소요 |
|---|---|---|---|---|
| T1 | 행렬과 자료구조 | 통과 | 17/17 | 0.12s |
| T2 | 수치 재구성 | 통과 | 9/9 | 0.16s |
| T3 | 전달 계수와 지연 | 통과 | 7/7 | 0.17s |
| T4 | 동기식 실행과 초기화 | 통과 | 10/10 | 0.34s |
| T5 | 학습 부호와 0교정 | 통과 | 16/16 | 0.17s |
| T6 | 작은 실현 가능한 학습 예제 | 통과 | 3/3 | 0.00s |
| T7 | 측정의 비침습성 | 통과 | 7/7 | 0.58s |
| T8 | 교사와 데이터 분리 | 통과 | 9/9 | 0.61s |
| T9 | 모델 저장·복원 | 통과 | 6/6 | 0.53s |
| T10 | 입력 규모 | 통과 | 10/10 | 3.64s |

- 전체: **통과 10 / 실패 0** (총 10개, 6.3s)
- A/B/C→D 예제 재현: E=0.8000, I=0.6000, u=0.2000, q=0 (기대값 0.8 / 0.6 / 0.2 / 0 과 일치). 이 결과를 근거로 C 가 잘못된 연결이라고 단정하지 않는다.
- 작은 학습 예제(T6): 6 반복에 해결, 흥분 가중치 0.600, 억제 가중치 0.100. 이 작은 예제의 성공은 실제 영상 과제의 성공이 아니다.

| 입력 크기 | 표본 수 | 유효 시야 비율 | 최대 sigma_pool | 최대 수용장 반경(px) | DoG 부호화 | 격자 표본 추출 |
|---|---|---|---|---|---|---|
| 64x64 | 128 | 75.6% | 5.12 | 10.25 | 0.3 ms | 2.0 ms |
| 1000x1000 | 128 | 78.4% | 125.05 | 250.10 | 48.5 ms | 3263.6 ms |

- 1000x1000 입력을 같은 전처리 API 로 처리했다. 이것은 입력 처리 시간이며 100만 뉴런 규모의 학습을 수행한 것이 아니다. 학습은 64x64 / 1024 뉴런 회로에서만 했다.

## 4. 데이터

| 분할 | 표본 수 | 가로/세로/빈 화면 | 밝은 선/어두운 선 | 대비 범위(진폭) | 선 중심 범위(px) |
|---|---|---|---|---|---|
| train | 240 | 38.3% / 39.2% / 22.5% | 90 / 96 | [0.252, 0.449] | x[12.2,53.1] y[10.4,52.6] |
| dev | 80 | 31.2% / 48.8% / 20.0% | 35 / 29 | [0.250, 0.447] | x[9.8,52.5] y[9.8,49.6] |
| test | 160 | 41.9% / 36.2% / 21.9% | 66 / 59 | [0.251, 0.450] | x[10.8,53.1] y[11.5,53.5] |
| test_novel | 160 | 30.6% / 48.1% / 21.2% | 67 / 59 | [0.461, 0.599] | x[3.0,58.9] y[3.0,61.1] |

- 대비는 배경 0.5 대비 **진폭**이다 (Michelson 대비 = 진폭/0.5). 약한 대비는 기본 데이터에서 제외했다.
- `test_novel` 은 학습률 선택에 쓰이지 않는 **독립 추가 시험**이며, 대비와 선 중심 반경 범위가 기본 시험과 다르다.
- 분할 간 `scene_id` 와 `base_scene_id` 가 겹치지 않음은 T8 에서 검사했다.

- L4 입력 정규화 계수: 훈련 표본의 양수값 95백분위 = **0.03904**, degenerate=아니오 (양수값 27005개). 검증·시험으로 재추정하지 않았다.
- L4 이진화 후 장면당 평균 활성 L4 뉴런 수 **47.27 / 256**, 활성 L4 가 하나도 없는 장면 16개(dev). L4 의 이진화로 소실된 정보를 뒤의 교사가 복구해 준다고 가정하지 않는다.

## 5. 대조 실험 결과 (모든 최종 평가는 **교정 없이** 수행)

| 조건 | 학습률 | test balanced acc | test F1 | test 비트 정확도 | dev balanced acc | 추가시험 balanced acc |
|---|---|---|---|---|---|---|
| frozen | 0.05 | 0.5004 (시드 1개) | 0.0016 (시드 1개) | 0.9704 (시드 1개) | 0.5000 (시드 1개) | 0.5000 (시드 1개) |
| correction_only | 0.05 | 0.5004 (시드 1개) | 0.0016 (시드 1개) | 0.9704 (시드 1개) | 0.5000 (시드 1개) | 0.5000 (시드 1개) |
| local_learning | 0.05 | 0.7058 (시드 1개) | 0.4342 (시드 1개) | 0.9670 (시드 1개) | 0.6949 (시드 1개) | 0.6716 (시드 1개) |
| shuffled_teacher | 0.05 | 0.5222 (시드 1개) | 0.0782 (시드 1개) | 0.9615 (시드 1개) | 0.5317 (시드 1개) | 0.5201 (시드 1개) |

주 지표는 balanced accuracy 와 F1 이다. 불활성 뉴런이 많으므로 전체 비트 정확도만으로 성공을 주장하지 않는다.

### 5.1 기준값 (항상 비활성 / 항상 발화)

| 기준 예측 | balanced accuracy | F1 | 비트 정확도 |
|---|---|---|---|
| 항상 비활성 | 0.5000 | 0.0000 | 0.9705 |
| 항상 발화 | 0.5000 | 0.0574 | 0.0295 |

- 시험 목표의 양성 비율은 **2.95%** 다.

### 5.2 혼동 행렬과 오류율 (조건·시드별 원자료)

| 조건 | 시드 | TP | FP | TN | FN | FPR | FNR | balanced acc | F1 |
|---|---|---|---|---|---|---|---|---|---|
| frozen | 42 | 1 | 5 | 39745 | 1209 | 0.0001 | 0.9992 | 0.5004 | 0.0016 |
| correction_only | 42 | 1 | 5 | 39745 | 1209 | 0.0001 | 0.9992 | 0.5004 | 0.0016 |
| local_learning | 42 | 518 | 658 | 39092 | 692 | 0.0166 | 0.5719 | 0.7058 | 0.4342 |
| shuffled_teacher | 42 | 67 | 436 | 39314 | 1143 | 0.0110 | 0.9446 | 0.5222 | 0.0782 |

### 5.3 frozen 대비 짝지은 차이 (test balanced accuracy)

| 조건 | 시드 수 | 평균 차이 | 표준편차 | 시드별 차이 |
|---|---|---|---|---|
| correction_only | 1 | 0.0000 | 시드 1개 | +0.0000 |
| local_learning | 1 | 0.2054 | 시드 1개 | +0.2054 |
| shuffled_teacher | 1 | 0.0219 | 시드 1개 | +0.0219 |

### 5.4 빈 화면 불필요 발화율 / E·I 평균 / 가중치

| 조건 | 빈 화면 발화율 | L2 E 평균 | L2 I 평균 | 학습 연결 가중치 L2 노름 | clipping 비율 |
|---|---|---|---|---|---|
| frozen | 0.00000 (시드 1개) | 0.3134 (시드 1개) | 0.3113 (시드 1개) | 7.389 (시드 1개) | 정의되지 않음 |
| correction_only | 0.00000 (시드 1개) | 0.3134 (시드 1개) | 0.3113 (시드 1개) | 7.389 (시드 1개) | 정의되지 않음 |
| local_learning | 0.00000 (시드 1개) | 0.3669 (시드 1개) | 0.2631 (시드 1개) | 7.465 (시드 1개) | 0.01533 (시드 1개) |
| shuffled_teacher | 0.00000 (시드 1개) | 0.3597 (시드 1개) | 0.2758 (시드 1개) | 7.627 (시드 1개) | 0.01819 (시드 1개) |

### 5.5 뉴런별 상태 (미관측 채널 / 지표 미정의)

| 조건 | 한 번도 발화 안 한 L2 수 | 양성 목표가 없는 L2 수 | 지표 미정의 L2 수 |
|---|---|---|---|
| frozen | 253.0 (시드 1개) | 3.0 (시드 1개) | 3.0 (시드 1개) |
| correction_only | 253.0 (시드 1개) | 3.0 (시드 1개) | 3.0 (시드 1개) |
| local_learning | 45.0 (시드 1개) | 3.0 (시드 1개) | 3.0 (시드 1개) |
| shuffled_teacher | 121.0 (시드 1개) | 3.0 (시드 1개) | 3.0 (시드 1개) |

- 양성 또는 음성 표본이 한 종류도 없는 뉴런은 balanced accuracy 가 정의되지 않아 NaN 으로 두었고 0 으로 채우지 않았다.
- 훈련 양성 표본 수(뉴런별): 최소 0, 최대 19, 평균 7.37; 양성이 한 번도 없는 채널 **4 / 256**. 훈련에서 양성 표본이 한 번도 없는 채널은 이 규칙으로 학습될 수 없다.

### 5.6 교정 중 성능과 교정 없는 성능 (반드시 분리)

| 조건 | dev (교정 켬, 참고용) | dev (교정 없음, 최종 평가 방식) |
|---|---|---|
| frozen | 정의되지 않음 | 0.5000 (시드 1개) |
| correction_only | 1.0000 (시드 1개) | 0.5000 (시드 1개) |
| local_learning | 1.0000 (시드 1개) | 0.6949 (시드 1개) |
| shuffled_teacher | 1.0000 (시드 1개) | 0.5317 (시드 1개) |

**유도 중 정확도는 최종 성능이 아니다.** 외부 교사가 목표 반응을 강하게 유도한 공학적 조작의 결과다.

### 5.7 위치·방향별 반응

**local_learning (seed=42) 의 방향별 시험 성능**

| 방향 채널 | balanced acc | F1 | TP | FP | FN |
|---|---|---|---|---|---|
| 선호 0° | 0.6922 | 0.4534 | 275 | 245 | 418 |
| 선호 90° | 0.7247 | 0.4143 | 243 | 413 | 274 |

**반경 bin 별 시험 성능 (안쪽 0 → 바깥 7)**

| 위치 | balanced acc | F1 | 양성 표본 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 반경 bin 0 | 0.4933 | 0.0000 | 120 | 0 | 67 | 120 |
| 반경 bin 1 | 0.5243 | 0.0645 | 99 | 7 | 111 | 92 |
| 반경 bin 2 | 0.6740 | 0.3776 | 103 | 37 | 56 | 66 |
| 반경 bin 3 | 0.6645 | 0.3246 | 107 | 37 | 84 | 70 |
| 반경 bin 4 | 0.6937 | 0.3920 | 145 | 59 | 97 | 86 |
| 반경 bin 5 | 0.7601 | 0.5343 | 209 | 113 | 101 | 96 |
| 반경 bin 6 | 0.8051 | 0.6479 | 257 | 161 | 79 | 96 |
| 반경 bin 7 | 0.7995 | 0.6172 | 170 | 104 | 63 | 66 |

### 5.8 같은 영상 재검사 · 새로운 위치·대비·극성

- **같은 영상 재검사**: 같은 초기 상태와 같은 입력으로 다시 실행하면 관찰 활동과 u 가 비트 단위로 동일하다 (T4 에서 검사). 다른 영상을 사이에 실행해도 같다 (상태 누출 없음).
- **새로운 위치·대비**: `test_novel` 은 대비 [0.46, 0.6] (기본 데이터 [0.25, 0.45] 밖), 선 중심 반경 [0.7, 0.95]×R (기본 0~0.7×R 밖) 로 만든 **독립 추가 시험**이다. 기본 시험과 분리해 정의했고 학습률 선택에 사용하지 않았다.

| 조건 | 기본 시험 balanced acc | 추가 시험(새 위치·대비) balanced acc |
|---|---|---|
| frozen | 0.5004 (시드 1개) | 0.5000 (시드 1개) |
| correction_only | 0.5004 (시드 1개) | 0.5000 (시드 1개) |
| local_learning | 0.7058 (시드 1개) | 0.6716 (시드 1개) |
| shuffled_teacher | 0.5222 (시드 1개) | 0.5201 (시드 1개) |

- **극성**: 밝은 선(+1)과 어두운 선(−1) 이 모든 분할에 함께 들어 있다 (4절 표). ON/OFF 두 채널이 모두 L4 로 들어가므로 학생은 두 극성을 같은 경로로 받는다.

## 6. 명세 [16] 의 질문에 대한 수치 답변

### Q1. 입력 목록과 당시 가중치로 발화 판단을 재구성할 수 있는가?

**예.** 진단 기록은 연결별 `received_value`, `weight_used`, `signed_contribution`, `weight_version` 을
모두 남기고, 그 합이 저장된 E/I/u 와 일치하는지 검사한다.
- 통과: E == 0.8 (관측 0.8)
- 통과: I == 0.6 (관측 0.6)
- 통과: u == 0.2 (관측 0.20000000000000007)
- 통과: q == 0 (관측 0)
- 통과: 저장한 연결 기여 합 == 저장한 E
- 통과: 저장한 연결 기여 합 == 저장한 I
- 통과: 재구성한 u == 저장한 u
- 통과: 도착 연결 3개 기록
- 통과: V1 L2 뉴런에서도 기여 합 == 저장한 E/I
- 기록된 상태 행 24개 / 연결 기여 행 788개 (최대 행 수 초과로 버린 행 0개). `diagnostics/trace_states.csv`, `diagnostics/trace_contributions.csv`.

예시 (`L2/ori0/k0/j0`, trial `diag/1`, tick 2, weight_version 917):

| 항목 | 저장값 | 연결 기여에서 재구성 |
|---|---|---|
| E | 1.802835 | 1.802835 |
| I | 1.300663 | 1.300663 |
| u | 0.502171 | 0.502171 |
| threshold | 0.500000 | - |
| q | 1 | - |
| 도착 연결 수 | 64 | - |

- 관측 사실과 원인 가설은 분리해 출력한다. 이 예시의 가설: (없음)
- 활동 판정: 활동 과다 (불필요한데 발화함, d=-1)

### Q2. 잘못 발화하거나 발화하지 못한 위치·방향 채널을 찾을 수 있는가?

**예.** 위치(반경 bin)·방향별 FP/FN 표(5.7절)와 장면별 불일치 지도 그림으로 찾는다.
- `local_learning` (seed=42) 기준 미발화(FN)가 가장 많은 위치: **radial_bin_0** (FN=120, 양성 120).
- 과다발화(FP)가 가장 많은 위치: **radial_bin_1** (FP=111).
- 측정된 방향 선호가 메타데이터 `preferred_orientation` 과 일치한 L2 채널 **156 / 256**, 측정 불가(반응 동률) 59개. 측정 방향 선호와 미리 지정한 preferred_orientation 메타데이터는 별개다.
- 그림: `figures/activity_after_seed*.png` (활동/목표/불일치 3단), `figures/orientation_preference_seed*.png`.

### Q3. 국소 가중치 학습 후 교사를 제거해도 개선이 유지되는가?

**유지된다.** 최종 시험 평가는 교사·교정을 모두 제거하고 일시 상태를 초기화한 자유 실행이다.
- frozen: 0.5004 (시드 1개) → local_learning: 0.7058 (시드 1개)
- 짝지은 차이 평균 **0.2054** (시드별 +0.2054)
- F1: frozen 0.0016 (시드 1개) → local_learning 0.4342 (시드 1개)
- 독립 추가 시험(새 대비·새 위치 범위)에서도: frozen 0.5000 (시드 1개) → local_learning 0.6716 (시드 1개)
- 저장한 체크포인트를 다시 불러와 교사 없이 예측했을 때 결과가 일치함은 T9 에서 검사했다.

### Q4. 활동 교정만 한 조건과 지속적 학습 조건이 구별되는가?

**구별된다.**
- `correction_only` 는 훈련 중 교정을 시연해 dev(교정 켬) 1.0000 (시드 1개) 를 보이지만,
  교정을 제거한 최종 시험에서는 0.5004 (시드 1개) 로 frozen(0.5004 (시드 1개)) 과 같다.
- frozen 과 correction_only 의 교정 없는 시험 예측이 **완전히 동일한가**: 예 (시드별 [True]).
  교정을 제거하고 일시 상태를 올바르게 초기화하면 두 조건의 반응은 같아야 한다. 같지 않다면 지속 상태 누출 가능성을 먼저 조사한다.
- `local_learning` 은 교정을 제거한 뒤에도 0.7058 (시드 1개) 를 유지한다.
- 가중치 변화 여부: correction_only 의 최종 가중치가 초기값과 비트 단위로 동일한가 = 예, local_learning = 아니오.

### Q5. 잘못된 교사, 정보 손실, 입력 부재, clipping 이 어떤 실패를 만드는가?

**잘못된 교사(shuffled_teacher)**: 훈련 목표의 전체 빈도는 보존하고 영상-목표 대응만 깨뜨렸다.
- test balanced accuracy 0.5222 (시드 1개) 로 frozen(0.5004 (시드 1개)) 수준에 머문다. local_learning(0.7058 (시드 1개)) 과 뚜렷이 다르다.
- 가중치는 실제로 움직였다: 학습 연결 L2 노름 7.627 (시드 1개) (frozen 7.389 (시드 1개)), clipping 비율 0.01819 (시드 1개), 한 번도 발화하지 않은 L2 수 121.0 (frozen 253.0).
- 그 결과 오경보율(FPR)이 0.0110 로 frozen(0.0001) 보다 높다. 즉 **학습은 일어나되 목표와 맞지 않는 방향**이며, balanced accuracy 는 거의 개선되지 않는다.
- 빈 화면 불필요 발화율: shuffled_teacher 0.00000 (시드 1개), frozen 0.00000 (시드 1개), local_learning 0.00000 (시드 1개).

**정보 손실 / 식별 불가능성**: 같은 L4 발화 패턴에 서로 다른 목표가 붙은 dev 표본 그룹 0개 (표본 0개, 충돌 목표 비트 0개), 서로 다른 감각 패턴 65개. 같은 L4 발화 패턴에 서로 다른 목표가 붙은 경우. 전처리·이진화·수용장 제한으로 생기는 식별 불가능성 후보이며, 이 표본들은 학생이 원리적으로 구분할 수 없다.
- **입력 부재**: dev 장면 중 활성 L4 가 하나도 없는 장면 16개. 이런 장면에서는 도착 입력이 전부 0 이므로 규칙상 가중치 갱신이 0 이다 (이를 숨기려고 임계값이나 기저 입력을 자동 조정하지 않았다).
- 실제로 도착 입력이 0 인 L2 뉴런 기록 예: `L2/ori0/k0/j0` (가설: ['입력 부재: 이 시점에 도착한 연결 입력과 감각 주입이 모두 없다', '부족한 흥분: 억제가 없는데도 흥분 합이 임계값에 못 미친다']).

**clipping**: 원래 G 와 clipping 후 실제 변화량을 모두 집계했다 (시드 대표값은 첫 시드 기준 개수).

| 조건 | clipping 비율 | 가중치 0 에 붙은 연결 | 가중치 상한에 붙은 연결 | 학습 연결 수 |
|---|---|---|---|---|
| frozen | 정의되지 않음 | 0 | 0 | 16384 |
| correction_only | 정의되지 않음 | 0 | 0 | 16384 |
| local_learning | 0.01533 (시드 1개) | 148 | 0 | 16384 |
| shuffled_teacher | 0.01819 (시드 1개) | 220 | 0 | 16384 |

- clipping 은 가중치를 [0, weight_max] 로만 자르므로 **연결 종류(부호)를 뒤집지 않는다** (T5 에서 검사).
- 0 에 붙은 흥분 연결과 상한에 붙은 억제 연결은 그 방향으로 더 학습할 수 없는 포화 상태다.

### Q6. 현재 교사는 어떤 정답 정보를 이미 알고 있는가?

교사는 다음을 **이미 알고 있다**. 이것은 정보상의 가정이며 학습으로 얻은 것이 아니다.

1. 장면의 도형 종류, 유한 선분의 양끝 좌표, 방향(도), 대비, 극성, 선폭 (원본 영상 좌표계 기준).
2. 각 L2 뉴런의 수용장 중심과 반경, 그리고 그 뉴런에 지정된 선호 방향.
3. 위 둘로부터 사람이 설계한 규칙 `d_i ≤ rf_radius_i/2 이고 방향차 ≤ 15°` 로 만든 뉴런별 이진 목표.
4. 어느 뉴런에 어떤 교정값을 보낼지 (L1Relay 의 목적지 사상).

- 이 목표는 **사람이 설계한 위치·방향 지도**이며 실제 V1 정상 반응의 측정값이 아니다.
- 이 목표를 학습한 것은 사전에 지정한 표현을 학습한 것이다. 라벨 없이 뉴런 연결의 의미를 자동 추출한 것이 아니다.
- IT 의 최종 오차 하나에서 이 지도를 자동 역산한 기능은 이 프로젝트에 **없다**.
- 학생(감각 경로)은 이미지 픽셀만 받는다. 라벨 메타데이터만 바꿔도 전처리·감각 입력·학습 전 출력이 비트 단위로 같음은 T8 에서 검사했다.

## 7. 초기값 적절성과 학습 동역학

| 조건 | 시드 | 학습 전 L2 발화율(train) | 초기 가중치 평균 | 최종 가중치 평균 | 최종 L2 노름 | 훈련 시행 수 | weight_version |
|---|---|---|---|---|---|---|---|
| frozen | 42 | 0.000146 | 0.0500 | 0.0500 | 7.389 | 0 | 0 |
| correction_only | 42 | 0.000146 | 0.0500 | 0.0500 | 7.389 | 1200 | 0 |
| local_learning | 42 | 0.000146 | 0.0500 | 0.0504 | 7.465 | 1200 | 917 |
| shuffled_teacher | 42 | 0.000146 | 0.0500 | 0.0508 | 7.627 | 1200 | 819 |

- 초기 가중치 U(0, 0.1) 에서 학습 전 L2 발화율은 **0.0146%** 다. 즉 초기에는 거의 **모두 비활성**이다 (명세가 보고하라고 한 '초기값이 부적절해 거의 모두 발화하거나 모두 비활성화되는지'에 대한 답).
- 흥분/억제 두 경로가 같은 L4 발화값을 받으므로 실효 가중치는 (w_exc − w_inh) ∈ [−1, 1] 이고,
  초기에는 평균 0 근처라 임계값 0.5 를 넘지 못한다. 학습은 이 실효 가중치를 키우는 방향으로 진행된다.

**local_learning (seed=42) 의 에폭별 추이 (dev, 교정 없음)**

| 에폭 | dev balanced acc | 시행당 평균 비영 오차 뉴런 수 | 가중치 갱신 적용 횟수 | 가중치 L2 노름 |
|---|---|---|---|---|
| 0 | 0.5429 | 8.36 | 186 | 19.911 |
| 1 | 0.6090 | 9.29 | 186 | 19.920 |
| 2 | 0.6522 | 9.12 | 183 | 19.924 |
| 3 | 0.6782 | 8.64 | 184 | 19.925 |
| 4 | 0.6949 | 7.83 | 178 | 19.926 |

## 8. 진단: 기여와 개입

**A/B/C→D 예제의 반사실 개입**

| 상태 | E | I | u | q |
|---|---|---|---|---|
| 기준 (w_C=0.6) | 0.8000 | 0.6000 | 0.2000 | 0 |
| 개입 (w_C=0.2) | 0.8000 | 0.2000 | 0.6000 | 1 |

- C 의 가중치를 0.6 -> 0.2 로 바꾸면 u 가 0.2 -> 0.6 이 되어 발화가 바뀐다. 이는 해당 조작의 효과를 보여줄 뿐, C 가 생물학적으로 잘못된 연결이라는 증명이 아니다.

**실제 회로의 반사실 개입**: 연결 9654 (`L4/OFF/k5/j10` → `L2/ori0/k4/j9`, excitatory, weight 0.0699) 를 0 으로 바꾸면 관찰 L2 256개 중 **0개**의 발화가 바뀐다.
- 해당 시행의 가중치 버전을 복원한 무개입 재실행이 저장된 반응과 일치: 예. (일치하지 않으면 과거 원인에 대한 개입 결과로 보고하지 않는다.)

**여러 선 위치·방향에서 같은 연결의 기여와 개입 효과**

| 장면 | 종류 | 방향 | 도착값 | 부호 있는 기여 | q(기준) | q(개입) | Δu |
|---|---|---|---|---|---|---|---|
| dev-00000 | horizontal_line | 0° | 1.0000 | 0.0699 | 0 | 0 | -0.0699 |
| dev-00001 | vertical_line | 90° | 1.0000 | 0.0699 | 0 | 0 | -0.0699 |
| dev-00002 | vertical_line | 90° | 0.0000 | 0.0000 | 0 | 0 | 0.0000 |
| dev-00003 | vertical_line | 90° | 0.0000 | 0.0000 | 0 | 0 | 0.0000 |
| dev-00004 | vertical_line | 90° | 1.0000 | 0.0699 | 1 | 0 | -0.0699 |
| dev-00005 | vertical_line | 90° | 0.0000 | 0.0000 | 0 | 0 | 0.0000 |
| dev-00006 | vertical_line | 90° | 0.0000 | 0.0000 | 0 | 0 | 0.0000 |
| dev-00007 | horizontal_line | 0° | 0.0000 | 0.0000 | 0 | 0 | 0.0000 |

- 여러 선 위치·방향에서 같은 연결의 기여와 개입 효과를 비교한 표다. 단일 가중치에 고정된 의미가 들어 있다고 가정하지 않는다.

- 진단 실행 전후로 학습 가중치가 비트 단위로 보존됨: 예 (T7 에서도 별도 검사).

## 9. 그림

- `results/quick/figures/activity_after_seed42.png`
- `results/quick/figures/activity_before_seed42.png`
- `results/quick/figures/activity_guided_seed42.png`
- `results/quick/figures/condition_summary.png`
- `results/quick/figures/diag_neuron_state_seed42.png`
- `results/quick/figures/input_mapping_seed42.png`
- `results/quick/figures/learning_curve_seed42.png`
- `results/quick/figures/orientation_preference_seed42.png`

## 10. 산출물 해시

### 코드·설정 (보고서 생성 시점에 다시 계산한 값)

| 파일 | 바이트 | sha256 |
|---|---|---|
| README_ko.md | 19474 | e0673f09e0333ddee5ca8cdc591f245a410c44db0ada34e05c4240d9deadfdfe |
| config.json | 5112 | 0f970e7f1137d12fa46a024bb95257d61cb80435e289ee0f45c37f60f6a2620f |
| make_manifest.py | 4571 | 7f0ded69f734f39ebbb16a3bb2845b47ba2e40394934e2244803065ceb1245a5 |
| make_report.py | 39497 | 3112bb9578f2504542b64b222b869a25cfb280d859e9415d7303924407335a83 |
| requirements.txt | 445 | 52e60c0d3fcae47be77f9ed1e4886e4231af356db388d03a868e04fb39eef424 |
| run_checks.py | 37497 | 9a1973c05e4ea2efa7e186fd96d9592b4a2e1f75430361aa950490c5196b1cb7 |
| run_demo.py | 9841 | 58739c0b7b12edddb134197d6fcaef6e5e78ff57362291b0a656b65f8fbfcfed |
| run_experiment.py | 32625 | 9a25b03072f6df9c281603555a6a0ca8691b47150ece5e8293f17cfbb6dbecd9 |
| src/__init__.py | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| src/connections.py | 10331 | 4024c8a9faed853a36f35ad23d8e998ed7aaa43cddfb85d18427557a71eb64cf |
| src/datasets.py | 10112 | 53f91c77933f66ba7713111db45040a60983586b95d2afe9e9642b9cb2f27176 |
| src/diagnostics.py | 17007 | b9e3dc4b996070dd912888a6732765a5bb592bd701bc6ae6d6c42efd10568c26 |
| src/engine.py | 15068 | 3ea32c29244dbfc4238a77cd8d860589aeb90f96ce1d0efabc38e338e9e42036 |
| src/events.py | 7715 | ca03ca7e0cf166ae93289f8548d14d016d396341db5b38d70b6017d37f2ead06 |
| src/experiment.py | 16950 | 199c4bb6c5405205cf3e02d98ca215fbb788a0e1cd98803d0c4075ec73b7d634 |
| src/local_learning.py | 8334 | db60eb419ada4d45aa6ac1d4fcc88a36a93ee063d9f5087411b4ae2f7f20bc4c |
| src/metrics.py | 7180 | dd55667da6e9fef48e1037199cdc3f9f98c608b3ea767a6a669ad9624415bdd9 |
| src/neuron.py | 16304 | 63dd6dadf1db42999d9c15a0e17d67a9b63447a059e1e22f3d792ffcf6926edc |
| src/persistence.py | 5410 | 67be75a399da22e831395ee8d74851aa7898caafff8a12841ff33559fa5407ee |
| src/protocols.py | 4501 | 348bdbb3ce16296a0ec5d612e6ac32e4d1bc6b1c70f9d05704e7ca42e01c07a2 |
| src/retina.py | 4749 | 7e889d8cad54356a0cad43f94f1fc45b2015a946fbe2b9465ed46c4ce17b022e |
| src/rngs.py | 1107 | 5e984b5983ef759df99f7328fa46b1bcd4abbfc1d1a61a0e94388566b1197e14 |
| src/spatial_mapping.py | 10394 | b584692b425dcba4761b8183b90c74bf55d1c711fcfdb5b936ebd889b5201ec0 |
| src/teacher.py | 8119 | d1c2760d77c7ba4ebff767a0a2f5de5789bfb578a0939cbfdd374e94c1d10899 |
| src/v1.py | 12599 | 371c9e7453645485cbfe698f61b54d6717b01985c8e52caa8da204e8156c148c |
| src/visualization.py | 10617 | 5e4cd42f0e4eaf8db63b14a4458098448dba6970cee4eb03fc0ce04e7a2a7188 |
| tests/test_units.py | 11422 | e103787dbc97223f08c92abec905c26bea2cff18813d95792ee5b76b4369e12a |

### 결과 파일 (실험 실행 종료 시점)

| 파일 | 바이트 | sha256 |
|---|---|---|
| results/quick/REPORT_ko.md | 26727 | e8545224952542264d4e02f3ecf35b7b9f653b5906785f313026104be9db1cc5 |
| results/quick/diagnostics/diagnostics.json | 43808 | e11c7592cfded22d86f27e74c4f5d2e20da7f9617024782caae14ea1dd51b28d |
| results/quick/diagnostics/trace_contributions.csv | 93122 | 40ab7618bf61170dc426e5a40c042ee14af559e1edee8bf78f12f0f93c8f4e1a |
| results/quick/diagnostics/trace_states.csv | 2688 | 49206b1072536f726f64da06f094d35ca93f04ae9971fd885bcf265e2397f720 |
| results/quick/figures/activity_after_seed42.png | 175333 | 01a39a1e8ab5646484dcc77469bc663459b40f9c4b22798dcde2e4603f2cb2a5 |
| results/quick/figures/activity_before_seed42.png | 174335 | a6e162404e9e81f75f4ce51e8af3d87d4e4ade7f25b9484262bdf8acbedf9e80 |
| results/quick/figures/activity_guided_seed42.png | 174991 | 93dc3e80453d562a5862727e9b585a854538b598dae26a9b6d0b68c6f59922a4 |
| results/quick/figures/condition_summary.png | 41324 | fd0cadd1ea7b4cd49ba1c16a3ea1648a5476fe3ccd15a544323a399f6442551c |
| results/quick/figures/diag_neuron_state_seed42.png | 57611 | eae3d225e35f50d407ab74fe961802a255de7d60fc3a78f6a5fe1366619906e2 |
| results/quick/figures/input_mapping_seed42.png | 129566 | 848cb119cedabc2d6c181b7bb85b35447efb08c0602451f86d3554a8bd07d5a9 |
| results/quick/figures/learning_curve_seed42.png | 47771 | d34b734fc526ce190299a5d0b8fd472798a6bb348b1125e9eb1b6882939459d4 |
| results/quick/figures/orientation_preference_seed42.png | 80170 | 1d667ca0361cc28ec8dad5799b7aa7fd77b17a6d25f01a2c5f1442de6e44376b |
| results/quick/metrics.csv | 1498 | d82cbb7c0e100d21db5a8a13048013de55efb685b1411d13ab63c1d93ffb258b |
| results/quick/models/correction_only_seed42.json | 553201 | 7b1ae438df93e9fd8c312f317d7a04984d32a177ee1b4f9b7dcf86b6a8b28309 |
| results/quick/models/correction_only_seed42.npz | 139469 | 407ce85800b4cb382ece5a604d498c6bfa1801f1656c12719199186c155d3659 |
| results/quick/models/frozen_seed42.json | 553192 | 697643063d8456183919cffb2f9c11f0f0bf933c506397e001b4367bed445807 |
| results/quick/models/frozen_seed42.npz | 139469 | 407ce85800b4cb382ece5a604d498c6bfa1801f1656c12719199186c155d3659 |
| results/quick/models/local_learning_seed42.json | 553202 | a14987653095f40d2845634cb05894e0a63934cbe31763d763def4587fe055ca |
| results/quick/models/local_learning_seed42.npz | 137787 | 8d67b0b9d50e90394940136eaa7e2df3e8809008e4ad1f3a94ea49e055f31fd5 |
| results/quick/models/shuffled_teacher_seed42.json | 553204 | 61a878c072f6d00dadc559f15623a781a258406ddedebd3387ea1a6cb2faaa43 |
| results/quick/models/shuffled_teacher_seed42.npz | 137604 | 50ddf8c184dd6964d72f6632e9defd72424e9d0dc33620a20b1e8e7bf42491ab |
| results/quick/raw/correction_only_seed42.json | 71151 | c83c073c0bea6f877a17f2b51131119c49fda7e712b6569c2ebb6f656013a20a |
| results/quick/raw/correction_only_seed42_predictions.npz | 4718 | 52fc36afcf3ada1454845a99f458bd5e0f0682026a61833e5ea1119a3d211dd0 |
| results/quick/raw/frozen_seed42.json | 69651 | a10d9297c12b39db0837ac160801618b7044da8122e0a4ce2f17a8026e61ef8f |
| results/quick/raw/frozen_seed42_predictions.npz | 4718 | 52fc36afcf3ada1454845a99f458bd5e0f0682026a61833e5ea1119a3d211dd0 |
| results/quick/raw/local_learning_seed42.json | 86678 | 1f3d6763290e72436a2129aabcaf6b338233ac03cc4c68b137bacf86222de595 |
| results/quick/raw/local_learning_seed42_predictions.npz | 7296 | 9717a6bec521abb17898dc3f62b490d0fa03dbc2f277079da9774528cc15683f |
| results/quick/raw/shuffled_teacher_seed42.json | 83098 | e9b0479aa9fda0b8365710a2299e09a7de70006ff0995b742eec60c6fdf8eadb |
| results/quick/raw/shuffled_teacher_seed42_predictions.npz | 6180 | 143669644ffee7101c1190f7cbfac22da4ff33cb6bf04d0053e23bd889761983 |
| results/quick/run_summary.json | 423037 | 96245f8aedca1c41a4fb70cffaf864a0ea441ef5812cbd5967817076baec255e |

- 실험 실행 종료 시점의 코드 해시 목록은 `run_summary.json` 의 `code_hashes` 에 그대로 남아 있다.
- 보고서·압축 파일까지 포함한 최종 해시 목록은 `results/HASHES.md` 와 `results/manifest.json` 에 있다 (`python make_manifest.py` 로 마지막에 생성). ZIP 자신의 해시는 ZIP 밖(같은 파일)에 기록한다.

## 11. 실제 실행한 것과 미실행 항목 (명확히 구분)

### 실제 실행한 것

- 필수 검증 T1~T10: 실행 (results/checks.json)
- 부가 단위 검사 `tests/test_units.py`: `python -m unittest discover -s tests` 로 실행 (37개 통과)
- 대조 실험 ['frozen', 'correction_only', 'local_learning', 'shuffled_teacher'] × 시드 [42] × 5 에폭: 실행 (총 20.2초)
- 조건·시드별 원자료 JSON/NPZ, metrics.csv, 모델 체크포인트, 그림, 진단 기록: 생성됨
- 저장 모델을 다시 불러와 교사 없는 추론·진단 데모: `run_demo.py --headless` 로 실행 (`results/demo/`)
- 학습률 탐색: 이 프로필에서는 고정 학습률을 썼고, 별도로 `python run_experiment.py --profile quick --lr-search` 를 **실제 실행**했다 (`results/quick_lrsearch/`).
  - `local_learning`: 후보 [0.01, 0.05, 0.1] → 평균 dev balanced accuracy {'0.01': 0.5432, '0.05': 0.6949, '0.1': 0.7089} → 선택 **lr=0.1** (여러 시드 평균 dev balanced accuracy 최대, 동률이면 작은 학습률)
  - `shuffled_teacher`: 후보 [0.01, 0.05, 0.1] → 평균 dev balanced accuracy {'0.01': 0.5051, '0.05': 0.5317, '0.1': 0.5278} → 선택 **lr=0.05** (여러 시드 평균 dev balanced accuracy 최대, 동률이면 작은 학습률)

### 미실행·한계 항목

- PyTorch / GPU: 설치하지 않았고 사용하지 않았다. 전부 NumPy/SciPy CPU 실행이다. 필수 의존성이 아니므로 설치를 반복 시도하지 않았다.
- V2 / V4 / IT / L5 / L6: 구현하지 않았다. 입출력 프로토콜과 설계 설명만 제공하며 호출하면 NotImplementedError 를 낸다 (src/protocols.py).
- 100만 화소(1000x1000) 입력은 전처리 API 로 **처리만** 했다. 그 규모의 회로를 구성하거나 학습하지 않았다. 학습은 64x64 입력 / 뉴런 1024개 / 연결 17408개 회로에서만 했다.
- 학습률 탐색: 이 실행에서는 **미실행**이다 (고정 학습률 0.05 모드). `--lr-search` 옵션으로 사전 고정 후보 [0.01,0.05,0.1] 탐색을 실행할 수 있다.
- 대화형 GUI 데모(run_demo.py 의 Matplotlib 슬라이더): 이 실행 환경에 디스플레이가 없어 **미실행**이다. 같은 추론 경로를 `--headless` 로 실행해 고정 장면 PNG 를 생성했다.
- 0°/90° 외의 방향 채널, 가로선/세로선 외의 도형, 색채 경로, 확률적 입력: 설정으로 확장 가능하지만 이번 실행에서는 **미실행**이다.
- 시험 세트는 선택 완료 후 조건·시드당 1회만 평가했고 모든 시험 지표는 저장된 동일 예측 배열 (raw/*_predictions.npz) 에서 계산했다. 시험 결과를 보고 구현이나 후보를 바꾸지 않았다.

## 12. 해석 시 주의

- 이 결과는 **사람이 설계한 위치·방향 목표 지도**를 학습한 결과다. 실제 V1 의 측정된 정상 반응이 아니다.
- 100만 화소 입력을 같은 API 로 처리한 것은 입력 처리일 뿐, 100만 뉴런 학습 완료가 아니다.
- L3 출력에 완전한 위상 불변성이나 고도화된 도형 인식을 주장하지 않는다.
- 억제 중계는 L4 발화를 한 tick 뒤에 전달하는 단순화이며 실제 억제성 세포의 다양성을 재현한 것이 아니다.
- 이 회로의 수치를 과거 DTP/KP 등 다른 구조의 실험 수치와 직접 비교하지 않는다.
- 기록 구조의 정확성(T1~T4, T7, T9), 활동의 일시적 교정(correction_only), 지속적 학습 효과(local_learning)
  는 서로 다른 검증 대상이며 위에서 각각 구분해 보고했다.
