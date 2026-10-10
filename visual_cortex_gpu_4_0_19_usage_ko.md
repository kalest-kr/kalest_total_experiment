# visual_cortex_gpu_4_0_19_local_correction.py 사용 안내

기준 파일은 `visual_cortex_gpu_4_0_18_single_model.py` (4.0.18) 이다. 첨부 파일의 SHA256 은 `fdc72de5225fa7d27ebaa0c15edb07c7d306bbfd1b11906d1ca63a24d4906c30` 이다.
바뀐 곳은 메뉴 27 (`--mode capacity_rebuild`) 의 교정 경로, 판독 창, 기록뿐이다. 메뉴 0~26 은 4.0.18 과 같다.

> L1·L5·L6 에 준 역할 (수신·검증, 하위 수정 후보 배분, amount/pattern 분해) 은 이 프로젝트의 계산 가정이다.
> 실제 뇌에서 확정된 오차 처리 알고리즘이 아니다.

---

## 1. 그대로인 것

- 한국어 메뉴, 데이터 로더 (`shapes/` 또는 `archive.zip`), 저장 폴더 처리, GPU 우선 장치 선택이 같다.
- 기본으로 최신 단일 모델 하나만 학습한다.
- 다음 회로 요소는 바꾸지 않았다.
  - 순방향 배선, 뉴런 수, 망막/LGN 전처리
  - 막전위 적분, 전달 지연, 시냅스 가중치, 전달 계수
  - L2 -> L3 일대일 대응 연결
- 다음 학습 규칙과 수치도 같다.
  - 목표 Q 생성 (SLSQP) 과 코사인 판정
  - `eta_commit = 0.2`, `guide_gamma = 0.1`, 목표 회전 상한 None
  - kappa (L2·L3) 0.25 mV, 패스 수 K = 2, error_scale 0.1
  - theta_fast 한계 ±1 mV, theta 초기값 3, 범위 1~30 mV
- 임계값 제어기 `DBThresholdController` 의 fast·commit 식도 그대로다.

## 2. 바뀐 것

### 2.1 공통 판독 창 계약

- 모든 영역·층이 같은 창을 쓴다. 대상은 L2/L3 활동과 L4 입력 발화율이다.
- 창은 `stimulus + offset` 관측 길이에서 계산한다. 현재 tick 1..T, 0~460 ms 이다.
  - 코드에 숫자로 박아 두지 않았다.
  - `lc_window_id(obs_steps, dt)` 와 `lc_readout_contract(...)` 가 창과 계약 해시를 만든다.
- 4.0.18 의 하위층·IT L2 판독 창 (64~160 ms) 은 교정에 쓰지 않는다.
- 판독 계약이 바뀌었으므로 이전 캐시는 재사용하지 않는다.
  - 4.0.17/4.0.18 C 준비 캐시는 다음 세 단계에서 거부된다.
    - `rw_contract_issues` 의 계약 해시 비교
    - `LCCalibrationRunner.load_state_dict` 의 `local_correction` 절 확인
    - `cr_check_c_source` 의 모드 비교
  - 거부되면 27-5 로 C 를 다시 준비해야 한다.
- 정규화 척도 s 는 새 창의 train 발화 수에서 다시 맞춘다.

### 2.2 교정 사슬 (기본 `correction.mode = "local_l5l6"`)

신호는 아래 순서로 전달된다.

```
비교기 L6 -> IT L1 -> IT L5 -> IT L6 -> V4 L1 -> V4 L5 -> V4 L6 -> V2 L1 -> ... -> V1 L1
```

- 기존 `ag_transfer` 는 이 경로에서 한 번도 부르지 않는다. 이것은 수용장 중첩으로 맞춘 D_A/T_B LMS 복원기다.
  - `ag_maps` 를 부르면 `LCPathError` 가 난다.
  - 학습 중 `LC_AG_TRANSFER_CALLS` 로 호출 수를 세어 Q9 로 기록한다.
  - 복원기 적합 단계 (D_A, T_B) 도 이 모드에서는 건너뛴다.
- 4.0.18 경로는 `correction.mode = "restorer_ag_transfer"` 를 명시했을 때만 돈다.
  - 그 경우 판독 창도 4.0.18 과 같다.

#### L5 (`lc_l5_propose`): 하위 L3 수정 후보와 제안량을 배분 근사로 계산

- 실제 활성 edge 표를 쓴다. 경로는 하위 L3 -> 상위 L4 -> 상위 L2 -> 상위 L3 이다.
  - gate_ref 의 활성 edge, 가중치 w, 지연, 시간 안 도달 여부가 들어 있다.
- 준비 단계에서 train 발화 수로 정적 이득을 잰다. 이득은 prior 표본으로 수축한 비율이다.
  - ρ23 = r3/r2
  - γ2 = r2/D
  - ρ34 = r4/r3_low
- 계산 순서 (상위 영역 U, 그 아래 영역 L):
  1. `dr3 = max(δ·s3, −r3)`
     - 요청 δ 를 Hz 로 바꾼다. 현재 발화율보다 많이 내리지 않는다.
  2. `dr2[p23[k]] = dr3 / ρ23`
     - 일대일 슬롯 k 에만 보낸다.
  3. `ΔD = dr2 / γ2`
  4. edge 별 요청
     - 양수이면 `ΔD / G` 이다.
     - 음수이면 `ΔD · r4 / D` 이다. 이미 활동하는 입력만 줄인다.
  5. `dr4` = 같은 L4 에서 나가는 edge 요청의 가중 평균 (Σ w·u / Σ w)
  6. `dr3_low = dr4 / ρ34`
  7. `δ_low = clip(h_low + dr3_low / s3_low, 0, 1) − h_low`
- 입력으로 x_A (r4), h_A2 (r2), h_A3 (h3, r3), t_A3, 연결·지연, 하위 상태를 모두 쓴다.
  - 경로에 없는 하위 슬롯은 후보가 되지 않는다 (Q4).
- 이 계산은 역함수도, 정확한 기울기도 아니다. 정적 이득으로 요청을 배분하는 근사다.

#### L6 (`lc_l6_decompose`): 하위 좌표계에서 분해

- 분해 식:
  - `amount = (h·δ / h·h) · h`
  - `pattern = δ − amount`
- 두 성분은 계산상 대리 지표다. amount 는 활동 크기 방향, pattern 은 그에 직교하는 방향이다.
- 분담 배율을 곱한 몫은 이 영역이 쓴다 (local). 나머지는 relay 로 아래 L5 에 보낸다.
- 상태 코드 (`LC_DECOMP_CODES`):

| 코드 | 이름 | 뜻 |
|---|---|---|
| 0 | decomposed | 정상 분해 |
| 1 | bootstrap_zero_activity | ‖h‖ ≤ eps 라서 방향이 없다. δ 전체를 pattern 으로 둔다 |
| 2 | zero | δ ≈ 0 이다. 패킷을 만들지 않는다 |
| 3 | unresolved_nonfinite | 비유한 값이다. 0 과 구별해 기록하고 패킷을 보내지 않는다 |

- 두 성분이 모두 ≈ 0 이면 패킷이 없다.
- 층별 변환은 `lc_l6_to_layers` 가 한다.
  - pattern 은 L2 로 간다. 일대일 슬롯에 ρ23 과 s3/s2 로 단위를 바꾼다.
  - amount 는 L3 로 간다.

#### L1 (`LCL1Receiver`): 수신·검증·버퍼·배달

- 패킷 (`LCCorrectionPacket`) 마다 다음을 확인한다.
  - 영역, 층, 뉴런 ID 순서, 표현 ID, 관측 창, 단위
  - 경로 지원 ID, sample / episode / round / state_version
  - 하나라도 맞지 않으면 거부한다 (Q6).
- attention 은 별도 필드로만 둔다. 검증에는 쓰지 않는다.
- pattern, amount, relay 는 따로 버퍼에 담는다.
- relay 는 임계값으로 배달하지 않는다. 그렇게 하려 하면 예외가 난다. relay 는 아래 L5 의 입력이 된다.
- 부호 규약 (전 경로 통일): δ > 0 이면 임계값을 내린다.
  - 식은 기존 `tl_threshold_delta(−δ)` 의 `κ·tanh(−δ/0.1)` 이다.

#### 영역 분담 `share_rule`

- `equal_split` (기본)
  - 영역별 local 비율은 IT 1/4, V4 1/3, V2 1/2, V1 1 이다.
  - 결과적으로 네 영역이 상위 요청의 1/4 씩 담당한다.
  - 이것은 공학적 선택이다. 최적이라는 근거는 없다.
- `full_each`
  - 각 영역이 전량을 담당한다. 진단용이다.

#### 동기 일정 (한 episode)

```
free -> t_app (1회, episode 동안 고정)
     -> K × [ pattern 패스 (L2 증분) -> 재실행 -> amount 패스 (L3 증분) -> 재실행 ]
     -> commit -> post-free
```

- 한 패스 안에서 모든 영역은 같은 settled 상태로 계산한다.
- settle 수는 1 + 2K + 1 로 고정이다. 조기 종료가 없다.
- 정답 라벨은 최상위 목표 방향을 고르는 데만 쓴다.

### 2.3 역전파 사용 범위

- 교정 신호를 만들 때 시뮬레이터 전체의 역전파·BPTT·autograd 를 쓰지 않는다.
- 미분이 쓰이는 곳은 기존 목표 Q 생성 (SLSQP) 하나뿐이며, 4.0.18 과 같다.
- L5 의 정적 이득은 발화 수 비율이다. 적합이나 기울기 계산이 없다.
- 그래도 L5 는 상위 요청을 실제 연결을 따라 아래로 보내는 계산이다. 성격상 역방향 신용 배분의 근사이며, "역전파가 아니다" 라고 주장하지 않는다.

---

## 3. 실행 순서 (Windows / PyCharm)

1. 이 파일 옆에 `shapes/` (circle, square, star, triangle 폴더) 또는 `archive.zip` 을 둔다.
2. PyCharm 의 Python 인터프리터에 numpy, scipy, torch (설치된 CUDA 판), pillow 가 있어야 한다.
   - matplotlib 과 h5py 는 선택이다.
   - 프로그램은 패키지를 자동 설치하지 않고 외부 서버에도 접속하지 않는다.
3. 실행하고 시작 질문에 답한다.

   ```
   python visual_cortex_gpu_4_0_19_local_correction.py
   ```

   - 저장 폴더를 붙여넣는다 (예: `D:\CortexResults`).
   - preset 은 아무 값이나 넣는다.
   - 장치는 `auto` / `cuda` / `cpu` 중에서 고른다.
   - 메뉴에서 **27** 을 고른다.
4. 메뉴 27 안에서 아래 순서로 실행한다.

| 순서 | 메뉴 | 내용 |
|---|---|---|
| 1 | **1** | A 구조 |
| 2 | **2** | B 전달 |
| 3 | **5** | C 준비. 새 공통 창 계약, 정규화 척도, 국소 이득 ρ23·γ2·ρ34 적합, routing_contract 기록 |
| 4 | **12** | 국소 교정 검사 Q1~Q10. C 폴더를 주면 그 준비를 재사용하고, 비우면 작은 표본으로 새로 준비한다 |
| 5 | **6** | D 학습 (최신 단일 모델 하나) |
| 선택 | **13** | 기능 검사 (합성 자극 40 종). 선, 격자, 반전, 대비, 이동, 모서리·십자·두 선 |
| 선택 | **14** | 작은 개입 진단. D 의 추론 체크포인트 또는 C 의 초기 임계값과 Q_0 를 쓴다 |
| 선택 | **15** | 오프라인 재집계와 REPORT_ko.md 재작성 |
| 선택 | **9 / 10** | 최종 test (정식 D) / 추론 |

- **S** (설정) 에서 다음 값을 바꿀 수 있다.
  - 교정 모드
  - 분담 규칙
  - 에폭당 snapshot episode 수
  - 추적 상세도 (`summary` / `full`)
- 메뉴 6 은 C 의 구성·표본 설정에, 현재 S 설정의 다음 값을 얹어 쓴다.
  - 교정 모드, 분담 규칙, snapshot 수, 추적 상세도
  - 모드가 C 와 다르면 거부된다.
  - 이득 prior (`gain_prior_samples`) 는 C 준비 때 고정되므로 C 값과 같아야 한다.
- 메뉴 7 (재개) 은 그 실행이 시작된 모드를 유지한다. correction 기록이 없는 4.0.17/4.0.18 D 는 `restorer_ag_transfer` 로 이어 간다.

### 명령줄 예

```
python visual_cortex_gpu_4_0_19_local_correction.py --mode capacity_rebuild --cr-task prepare  --cr-b-run "<B 폴더>" --device cuda --output "D:\CortexResults"
python visual_cortex_gpu_4_0_19_local_correction.py --mode capacity_rebuild --cr-task lc_check --cr-c-run "<C 폴더>" --device cuda --output "D:\CortexResults"
python visual_cortex_gpu_4_0_19_local_correction.py --mode capacity_rebuild --cr-task train    --cr-c-run "<C 폴더>" --device cuda --output "D:\CortexResults"
python visual_cortex_gpu_4_0_19_local_correction.py --mode capacity_rebuild --cr-task probe    --run-dir "<D 또는 C 폴더>" --cr-probe-n 4 --cr-probe-scale 1.0 --device cuda --output "D:\CortexResults"
python visual_cortex_gpu_4_0_19_local_correction.py --mode capacity_rebuild --cr-task functional --device cuda --output "D:\CortexResults"
python visual_cortex_gpu_4_0_19_local_correction.py --mode capacity_rebuild --cr-task analyze  --run-dir "<D 폴더>"
```

추가 인자:
- `--cr-correction-mode local_l5l6|restorer_ag_transfer`
- `--cr-share-rule equal_split|full_each`

### `correction` 설정 블록 (기본값)

| 키 | 기본값 | 뜻 |
|---|---|---|
| `mode` | `local_l5l6` | `restorer_ag_transfer` 이면 4.0.18 경로 |
| `share_rule` | `equal_split` | `full_each` 는 진단용 |
| `gain_prior_samples` | 1.0 | 이득 비율 수축 prior (C 준비 때 고정) |
| `rho_min` | 0.05 | 중계 이득 하한 (0 나눗셈 방지) |
| `decomposition_eps` | 1e-8 | ‖h‖ 가 이 값 이하이면 bootstrap |
| `zero_tol` | 1e-12 | 이 값 이하이면 zero 로 판정 |
| `snapshot_episodes_per_epoch` | 4 | 상세 활동 snapshot 수 (0 = 끔) |
| `trace_detail` | `summary` | `full` 이면 모든 행을 기록 (파일이 커진다) |

---

## 4. 출력 (D 학습 폴더 기준)

| 파일 | 내용 |
|---|---|
| `config.json`, `config_effective.json`, `manifest.json` | 설정, 실효 학습값, 판독 계약 해시, 파일 목록·해시 |
| `routing_contract.json` | 영역별 실제 edge 경로 요약, 시간 도달, 경로 지원 ID, 이득 요약, 공통 창 |
| `prechecks.json` | Q1~Q6 (순수 배열), 경로 검사, Q8 시작 해시. Q7·Q10 은 NOT_RUN (27-12 에서 실행). Q8 끝 해시와 Q9 는 학습 끝에 채운다 |
| `correction_trace.jsonl` | lc_l6 / lc_l5 / lc_l1 / commit / guided_output / buffer_integrity 행. `full` 이면 모든 행 |
| `activity_snapshots/part_XXXXX.npz` | 제한된 episode 의 패스별 L4 입력·L2/L3 활동·theta_fast·commit 증분 |
| `epoch_metrics.csv` | 기존 지표 (변경 없음, 공식 cosine_accuracy) |
| `correction_summary.npz` | 표본 열 (h_guided 패스별 각도·거리, 분해 코드 수, float32 손실 크기) 과 벡터 (h_pre, t_applied, h_post, h_guided_pass{k}) |
| `target_geometry.npz` | 버전별 Q |
| `lc_interference__<조건>.csv` | 표본 간 간섭. 고정 Q_ref (버전 V−2) 기준 self / between / net gain, 개입 표본 수, 클래스별 값 |
| `lc_geometry_by_epoch.csv` | 클래스 간/내부 기하, Q_epoch·Q_ref 마진, argmax 정답 개수와 표본 수 n, 목표 최소 각도. REPORT_ko.md 는 이를 `xx.xx% (정답/전체)` 로 적는다 |
| `lc_trace_summary.json`, `lc_analysis.json` | 추적 요약과 오프라인 분석 묶음 |
| 체크포인트 (`inference_checkpoint.npz/.json` 등) | 기존과 같다 |
| `REPORT_ko.md` | 한국어 보고서. 실행하지 않은 항목은 NOT_RUN 으로 표시 |

메뉴 13 은 `functional_tests.json` 과 `functional_responses.npz` 를 쓴다. 메뉴 14 는 `lc_probe.json` 과 `lc_probe.csv` 를 쓴다.

- 개입 진단 행에는 다음 값이 있다.
  - cos (영 벡터이면 undefined)
  - `actual_over_requested_norm`
  - `area_share_of_top_request`
- 평가하지 않은 confidence 는 확률 1.0 으로 쓰지 않는다.
- 이전 실행의 수치는 현재 실행 결과에 섞지 않는다.

---

## 5. 이전 경로와 새 경로 대응

| 단계 | 4.0.18 | 4.0.19 (local_l5l6) | 입력 | 출력 |
|---|---|---|---|---|
| IT 목표·교사 gate | `ag_targets` -> `ag_make_step` (γ 0.1, 회전 상한 None) | 같은 함수 (변경 없음) | free h_IT, 라벨, Q_e | t_app, gate (episode 동안 고정) |
| 판독 창 | IT L3 0~460 ms, 하위층·IT L2 64~160 ms | `LCPairedNetwork.win` = tick 1..T. 전 층 공통 (`lc_readout_contract`) | spike_count | h = clip((count/T)/s, 0, 1), L4 입력 발화율 r4 |
| 상위 L3 -> 하위 L3 | `ag_transfer`: D_A(h+d) − D_A(h) (수용장 중첩 LMS 복원기) | `lc_l5_propose` (L5): 실제 edge + 정적 이득의 배분 근사 | h3, t3, r2, r3, r4, 하위 h3/r3, 경로 표, 이득 | δ_low, 중간량 dr2·ΔD·dr4 |
| L3 -> L2 | `ag_transfer`: T_B(h3+d3) − T_B(h3) (이웃 L2 로 퍼짐) | `lc_l6_to_layers`: 일대일 슬롯 k 에만, ρ23 으로 단위 변환 | pattern 성분, ρ23, s3, s2 | δ_L2 |
| 교정 종류 | shape = L2, intensity = L3 (이름만) | `lc_l6_decompose` (L6): amount = (h·δ/h·h)h, pattern = δ − amount | 하위 좌표계 δ, h | amount -> L3, pattern -> L2, relay -> 하위 L5, 상태 코드 |
| 영역 간 분담 | 모든 영역이 같은 IT 요청을 각자 전량 담당 (명시 규칙 없음) | `equal_split`: IT 1/4, V4 1/3, V2 1/2, V1 1 | δ | local / relay |
| L1 수신 | `TLReceiver` (영역·층·sample·version·표현) | `LCL1Receiver`: 뉴런 ID 순서·관측 창·단위·경로 ID·episode 검사를 더함, 채널별 버퍼 | `LCCorrectionPacket` | 임계값 증분 κ·tanh(−δ/0.1) |
| 임계값·commit | `DBThresholdController` (fast ±1, eta 0.2) | 같은 객체 | 증분 | theta_fast, theta_base |

같은 표가 `config_effective.json` 의 `capacity_rebuild.local_correction.path_table` 과 REPORT_ko.md 에도 들어간다.

---

## 6. 검사 상태

### 이 파일을 만든 환경에서 실행한 것 (torch 없음)

- 정적 검사
  - `py_compile` 통과
  - `pyflakes` 는 4.0.18 부터 있던 경고 1 건 (`field` 가 반복 변수에 가려짐) 만 남는다.
  - AST 이름·서명 대조 통과
- NumPy 합성 입력 단위 검사 Q1~Q6: 모두 PASSED

| 검사 | 내용 |
|---|---|
| Q1 | 요청 0 이면 갱신 0 |
| Q2 | 부호와 상하한 |
| Q3 | L2 -> L3 일대일 슬롯 밖으로 새지 않음 |
| Q4 | 경로 밖 후보 없음 |
| Q5 | 분해 경우: 평행, 직교, 혼합, h = 0 bootstrap, δ = 0, NaN 미해결, 분담 |
| Q6 | 잘못된 패킷 거부 |

- 추가 합성 검사 20 개 통과
  - 이득 비율, L5 부호, r3 하한
  - 간섭 집계 정확값
  - 합성 실행 폴더의 오프라인 분석과 보고서
  - manifest, 기하 행, 자극 40 종
  - 계약 해시가 이전과 다름, 창 끝값
  - relay 배달 거부, 분담 비율
- C/D 소스 호환 검사 7 개 통과
  - D 단계 값만 다르면 허용
  - 4.0.18 C 거부
  - 이득 prior 불일치 거부
  - legacy 쌍 허용
  - 저장 설정의 모드 유지

이 검사들은 고정 규칙이 손실을 줄이거나 분류를 개선하는지 확인하지 않는다. 계산 경로·부호·마스크·검증·기록이 맞는지만 확인한다.

### NOT_RUN (사용자 PC 에서 실행)

- 실제 데이터셋으로 하는 A/B/C/D 실행. C 준비, D 학습, 최종 test, 추론이 여기에 든다.
- 모델 실행 검사 Q7~Q10 (메뉴 27-12)

| 검사 | 내용 |
|---|---|
| Q7 | 측정 순도 |
| Q8 | 순방향 가중치 불변 |
| Q9 | ag_transfer 호출 0 |
| Q10 | 반복성. GPU 비결정성 때문에 불일치하면 FAILED 가 아니라 MEASURED |

- 기능 검사 (27-13), 개입 진단 (27-14)
- torch 경로 실행 (LCPairedNetwork settle·chain·episode 의 GPU 계산)
- 메뉴 0~26 과 restorer_ag_transfer 모드의 회귀 실행, GPU 시간 측정

---

## 7. 한계와 해석 주의

- 이득 ρ23·γ2·ρ34 는 C 준비 때 고정한 평균 비율이다.
  - 학습 중 임계값이 바뀌어도 다시 맞추지 않는다.
  - 비선형 스파이크 응답의 정확한 역이 아니다.
- `equal_split` 은 공학적 선택이다. 영역 분담의 최적성이나 생물학적 근거는 없다.
- 이 국소 규칙이 손실이나 정확도를 개선한다는 보장은 없다. D 결과로 판단해야 한다.
- 4.0.19 는 교정 경로 (ag_transfer -> L5/L6/L1) 와 판독 창 (64~160 -> 전체 창) 을 함께 바꿨다.
  - 그래서 4.0.18 대비 차이를 어느 한쪽 변경 탓으로 돌릴 수 없다.
  - 나눠 보려면 `restorer_ag_transfer` 모드 결과와 함께 비교해야 한다. 다만 그 모드는 판독 창도 4.0.18 그대로다.
- amount/pattern 은 하위 좌표계의 기하 분해다. 계산상 대리 지표이며 "크기 신호 / 모양 신호" 의 생리적 실체를 뜻하지 않는다.
