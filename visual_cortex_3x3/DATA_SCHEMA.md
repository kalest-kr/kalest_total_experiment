# DATA_SCHEMA.md — 3×3 인터페이스와 이벤트·상태·결과 스키마

## 1. 3×3 뉴런 기록 인터페이스

`cortex.records.NeuronRecord.as_matrix()` 는 **object 배열** `(3, 3)` 을 돌려준다.
`float32[3,3]` 수치 행렬이 아니다.

| 사용자 표기 | Python 인덱스 | 내용 | 타입 | 단위 |
|---|---|---|---|---|
| 1행 1열 | `[0][0]` | x 좌표 | `float` | mm |
| 1행 2열 | `[0][1]` | y 좌표 | `float` | mm |
| 1행 3열 | `[0][2]` | z 좌표 | `float` | mm |
| 2행 1열 | `[1][0]` | **출력 연결 목록** | `OutgoingConnectionsView` | — |
| 2행 2열 | `[1][1]` | 발화 임계값 θ | `float` | mV (전도도 모드) / 무차원 (합산 모드) |
| 2행 3열 | `[1][2]` | 기저 출력 이득 P | `float` | 무차원 |
| 3행 1열 | `[2][0]` | 입력 이벤트 로그 뷰 | `InputLogView` | — |
| 3행 2열 | `[2][1]` | 동적 상태 참조 | `DynamicStateView` | — |
| 3행 3열 | `[2][2]` | 메타데이터 참조 | `MetadataView` | — |

**불변 규칙**

* `[1][0]` 에는 입력 총합을 넣지 않는다. 총합은 `Engine` 의 지역 변수이고,
  모니터링이 필요하면 `DynamicStateView.interval_activation` 또는 진단 기록을 본다.
  같은 값을 두 곳에서 진실의 원본으로 관리하지 않는다.
* `[2][0]` 의 로그는 **발화해도 삭제하지 않는다**. 현재 연산에 쓰는 작업 버퍼
  (`g_nS`, `interval_activation`)는 별도 배열이다.
* 3×3 뷰는 `NeuronArrays` 의 **같은 메모리**를 참조한다. 미러 복사본이 없다.
* 이웃 뉴런 객체를 재귀적으로 담지 않는다. 정수 ID 만 쓴다.
* `P` 는 출력 이득이다. 방출 확률이 필요하면 `release_probability` 로 따로 이름
  붙여야 한다 (이번 구현에는 없다).

### 뷰가 제공하는 것

```
OutgoingConnectionsView : .synapse_ids (int64 배열, slice 뷰), .rows(), .targets()
InputLogView            : .count(), .rows(t_from_ms, t_to_ms, limit), .last_consumed_event
DynamicStateView        : .V_mV (3,), .g_nS (3,R), .refractory_until_ms,
                          .last_spike_ms, .rate_estimate_hz, .trace_pre, .trace_post,
                          .interval_activation, .fired, .snapshot()
MetadataView            : .area, .layer, .cell_type, .dale, .compartments,
                          .visual_field_xy_deg, .receptive_field, .preferred, .to_dict()
```

---

## 2. 상태 배열 (`cortex.records.NeuronArrays`)

행 인덱스 = `neuron_id`. 모든 배열은 한 번 할당되고 shape 가 바뀌지 않는다.

| 배열 | shape | dtype | 의미 |
|---|---|---|---|
| `position_mm` | `(N,3)` | float64 | 전역 피질 좌표 |
| `surface_uv_mm` | `(N,2)` | float64 | 영역 표면 국소 좌표 |
| `apical_depth_mm` | `(N,)` | float64 | 첨단수상돌기 다발 깊이 (L1 접점) |
| `threshold` | `(N,)` | float64 | 3×3 `[1][1]` |
| `output_gain_P` | `(N,)` | float64 | 3×3 `[1][2]` |
| `area_id`, `layer_id`, `cell_type_id` | `(N,)` | int32 | 레지스트리 정수 |
| `dale_sign` | `(N,)` | int8 | +1 흥분 / −1 억제 |
| `has_compartment` | `(N,3)` | bool | 존재하는 구획 |
| `C_pF`, `gL_nS`, `EL_mV` | `(N,3)` | float64 | 구획 파라미터 |
| `g_couple_nS` | `(N,3)` | float64 | `[:,1]=soma-basal`, `[:,2]=soma-apical` |
| `V_mV` | `(N,3)` | float64 | 막전위 |
| `g_nS` | `(N,3,R)` | float64 | 구획×수용체 전도도 (비음수) |
| `refractory_until_ms`, `last_spike_ms` | `(N,)` | float64 | |
| `spike_count` | `(N,)` | int64 | |
| `rate_estimate_hz` | `(N,)` | float64 | 지수 이동 평균 |
| `trace_pre`, `trace_post` | `(N,)` | float64 | STDP 흔적 |
| `interval_activation` | `(N,)` | float64 | 합산 모드 **작업 버퍼** |
| `last_decision_value`, `last_decision_threshold` | `(N,)` | float64 | 판정 직전 상태 |
| `fired` | `(N,)` | int8 | |
| `last_consumed_event` | `(N,)` | int64 | 중복 주입 방지 위치 |
| `visual_field_xy_deg` | `(N,2)` | float64 | **시야 좌표 (deg)** — 피질 mm 와 다른 공간 |
| `rf_sigma_deg` | `(N,)` | float64 | |
| `pref_orientation_rad`, `pref_phase_rad` | `(N,)` | float64 | NaN 이면 해당 없음 |
| `ocular_dominance` | `(N,)` | float64 | −1..+1 |
| `eye_id`, `on_off` | `(N,)` | int8 | |
| `channel_id` | `(N,)` | int32 | 망막 채널 |
| `hypercolumn_uv` | `(N,2)` | float64 | hypercolumn 내부 미세 좌표 |
| `out_ptr`, `out_syn` | `(N+1,)`, `(K,)` | int64 | 출력 CSR 인덱스 |

---

## 3. 시냅스 edge table (`cortex.synapses.SynapseTable`)

행 인덱스 = `synapse_id`.

| 필드 | dtype | 의미 |
|---|---|---|
| `src_id`, `dst_id` | int32 | 발신/수신 뉴런 |
| `src_area`, `dst_area` | int16 | 영역 ID |
| `target_layer` | int16 | 표적 층 ID |
| `target_compartment` | int8 | 0=soma, 1=basal, 2=apical |
| `receptor_type` | int8 | 수용체 레지스트리 ID |
| `weight` | float64 | **비음수 크기.** 부호는 `src_dale_sign` |
| `weight_unit` | str (테이블 속성) | `nS` 또는 `dimensionless` |
| `base_delay_ms` | float64 | 연속 지연 |
| `effective_delay_steps` | int32 | 양자화 지연 (≥ `min_delay_steps`) |
| `plasticity_rule` | int8 | 0=none, 1=stdp |
| `plasticity_state_id` | int64 | 규칙별 상태 행 인덱스 (−1 이면 없음) |
| `active` | bool | 소거 실험에서 끌 수 있다 |
| `src_dale_sign` | int8 | +1/−1 |
| `rule_index` | int16 | 어느 배선 규칙에서 나왔는지 |

인덱스: `in_ptr/in_syn` (dst 기준 CSR), `out_ptr/out_syn` (src 기준 CSR).
`connectivity_matrix()` 는 시각화·집계용 `scipy.sparse` 행렬이며 계산 경로에
쓰지 않는다.

---

## 4. 이벤트 로그 (`events.h5` / `events.npz` 의 `events/` 그룹)

| 열 | dtype | 의미 |
|---|---|---|
| `event_id` | int64 | 실행 안에서 단조 증가. 체크포인트에서 이어진다 |
| `parent_spike_id` | int64 | 이 사건을 만든 발화의 ID |
| `sample_id` | int32 | 자극 표본 인덱스 |
| `episode_id` | int32 | 에피소드 (시퀀스) 인덱스 |
| `src_id` | int32 | 발신 뉴런 |
| `dst_id` | int32 | 수신 뉴런 (발화 사건은 −1) |
| `synapse_id` | int64 | 연결 (발화 사건은 −1) |
| `emit_time_ms` | float64 | 발신 시각 |
| `arrival_time_ms` | float64 | 도착 시각 |
| `arrival_step` | int64 | 도착 스텝 |
| `weight_snapshot` | float64 | **실제 적용한** 가중치 |
| `source_gain_snapshot` | float64 | **실제 적용한** P |
| `target_compartment` | int8 | 0/1/2, 발화 사건은 −1 |
| `event_type` | int8 | `EVENT_TYPES` 인덱스 |
| `amount` | float64 | 기여량 (합산 모드) 또는 Δg (전도도 모드, ≥0) |
| `amount_unit` | int8 | `AMOUNT_UNITS` 인덱스 |

`EVENT_TYPES = (external_drive, synaptic_arrival, spike, weight_update, threshold_update)`
`AMOUNT_UNITS = (dimensionless, nS, pA, spike, weight_delta, mV)`

행 1건은 약 88 바이트로 추정한다 (`estimate_sizes` 에서 사용).

**기록 모드**

| 모드 | 저장 내용 | 필수 항목 |
|---|---|---|
| `full` | 모든 이벤트 | — |
| `selected` | 선택 뉴런에 도착/발신한 이벤트만 | `selection_criterion` 필수 |
| `summary` | 개별 사건 없음, 종류별 개수만 | `selection_criterion` 필수 |

버려진 건수는 `EventLog.n_skipped_by_mode` 로 **세어서** manifest 에 남긴다.
조용히 버리고 `full` 이라고 표시하지 않는다.

---

## 5. 상태 기록 (`states.h5` 의 `states/` 그룹)

| 열 | dtype |
|---|---|
| `step` | int64 |
| `time_ms` | float64 |
| `sample_id` | int32 |
| `neuron_id` | int32 |
| `V_soma_mV`, `V_basal_mV`, `V_apical_mV` | float64 |
| `g_total_nS` | float64 |
| `threshold` | float64 |
| `rate_hz` | float64 |
| `fired` | int8 |
| `interval_activation` | float64 |

층별 집계는 `aggregate/layer_totals` 에 `name`(S64), `spike_count`(int64),
`mean_rate_hz`(float64) 로 저장된다.

---

## 6. 갱신 기록 (`updates.h5` 의 `updates/` 그룹)

`cortex.plasticity.UpdateRecord` 의 스칼라 필드가 열이 된다:
`step, time_ms, n_synapses_changed, ltp_sum, ltd_sum, decay_sum, clipped_low,
clipped_high, weight_delta_sum, weight_delta_abs_sum, theta_delta_sum,
theta_delta_abs_sum, n_theta_clipped, teacher_derived`.

**규칙별 항(LTP/LTD/감쇠)과 제한(clipping) 전후 변화량을 따로 기록**하므로,
교사 유래 항과 독립 항을 분리해 볼 수 있다.

---

## 7. manifest.json

```
run_id, status, started_utc, finished_utc, elapsed_seconds, command, resumed,
config (해석된 전체 설정), config_sha256,
code {combined_sha256, files{path: sha256}, n_files}, git_commit,
data_hashes, library_versions, platform, device, gpu_used, dtypes,
thread_settings, units, recording {mode, backend, selection_criterion, ...},
assumptions {species_assumption, note_ko}, experiment_status,
model {n_neurons, n_synapses, wiring, anatomy, synapse_summary},
input_normalization, splits, test_access {n_test_evaluations, note_ko},
plasticity_summary, event_log_schema, resumed_from,
n_state_rows_written, n_state_rows_skipped_by_limit,
transmission_headroom, silent_areas, stimulus_cap
```

`status ∈ {running, completed, interrupted, failed}`.
**이미 `completed` 인 디렉터리는 덮어쓰지 않는다** (`--run-dir` 를 새로 주어야 한다).

### 7.1 `input_normalization`

```
percentile            백분위 (기본 99)
source_split          추정에 쓴 분할 이름 (train / simulate_prefix / resume / ...)
fit_representation    "grid_samples" | "image_pixels"
                      normalize() 를 적용하는 표현과 같아야 한다
min_scale_ratio       채널 스케일의 바닥 = min_scale_ratio * max(scale)
floored_channels      바닥에 걸린 채널 번호 (신호가 거의 없는 채널)
n_images, scale, degenerate_channels
```

### 7.2 `transmission_headroom` (실행 전 진단)

흥분성 배선 규칙마다 **시냅스 전달이 임계에 닿을 수 있는지** 손계산한 결과다.
`conductance_lif` 모드에서만 채워진다 (`applicable: false` 면 계산하지 않음).

```
rules[]:
  rule                     규칙 이름
  n_synapses               이 규칙이 만든 시냅스 수
  mean_in_degree           표적 뉴런 1개당 평균 시냅스 수
  mean_weight_nS           평균 가중치
  g_need_nS                gL * (V_th - EL) / (E_rev - V_th)
  presyn_rate_needed_hz    g_need / (deg * w * tau/1000)
  presyn_rate_max_hz       앞 영역이 낼 수 있는 상한
                           (피질: 1000/t_ref, 망막: 외부 구동 상한)
  presyn_area              앞 영역 이름
  reachable                needed <= max
blocked_rules[]            reachable=false 인 규칙 이름
```

단일 구획 정상상태 근사이므로 정확한 예측이 아니라 **자릿수 점검**이다.
`blocked_rules` 가 비어 있지 않으면 그 뒤 영역은 어떤 입력에도 침묵할 수 있다.

### 7.3 `silent_areas` (실행 후 진단)

```
spikes_by_area_total   영역별 총 스파이크 수
silent_areas[]         모든 표본에서 한 번도 발화하지 않은 영역
active_areas[]         한 번이라도 발화한 영역
all_silent             전부 침묵했는가
```

`silent_areas` 가 비어 있지 않으면 그 영역의 0 은 **모형의 결론이 아니다.**
`transmission_headroom` 을 먼저 보라.

### 7.4 `stimulus_cap`

```
cap, n_generated, n_used, truncated
```

`experiment.max_stimuli` (또는 `run-all --limit-stimuli`) 로 자극을 잘랐는지.
조용히 줄이지 않고 항상 기록한다.

---

## 8. metrics.jsonl / summary.csv

한 줄이 하나의 측정이다. `kind` 가 종류를 구분한다.

* `kind="sample"` : `sample_index, stimulus_id, label, n_spikes_total,
  spikes_by_area, mean_rate_hz_by_area, first_spike_step_by_area,
  silent_fraction_by_area, n_steps, n_events`, 그리고 측정 조건
  (`split`, `learning`, `engine_mode`)
* `kind="validation"` : `n_checks, n_passed, n_failed, n_skipped`
* `kind="readout"` : `classifier, l2, train/dev/test_accuracy, ...`

`summary.csv` 는 같은 줄들을 표로 펼친 것이다 (중첩 값은 JSON 문자열).

---

## 9. 체크포인트 (`checkpoints/<name>.npz` + `.json`)

`.npz` 에는 `engine/<field>` 배열과 `queue/<i>/<field>` (대기 이벤트) 가 들어간다.
`.json` 에는 `sample_index, event_log_position, data_locator, config_sha256,
code_sha256, rng_state, queue_steps, engine_<scalar>` 가 들어간다.

재개할 때 `config_sha256` / `code_sha256` 를 비교한다. 다르면 오류를 내고,
`--allow-mismatch` 를 명시해야 진행한다. `event_log_position` 과 `event_id`
카운터를 복원하므로 **이벤트가 중복 기록되지 않는다**.

---

## 10. RNG 스트림

```
SeedSequence([master_seed, stream_offsets[name], sub_index])
```

`stream_offsets` 는 설정에 **고정 정수**로 선언된다
(`data, novel_data 없음 / wiring, weights, input_noise, learning_order,
diagnostics, readout, split, stimulus`). Python 의 실행별 `hash()` 에 의존하지
않으며, 새 스트림을 추가해도 기존 스트림의 난수열이 바뀌지 않는다.

---

## 11. 재현 범위

같은 `master_seed`, 같은 `config_sha256`, 같은 `code_sha256`, 같은 라이브러리
버전에서 다시 실행하면 같은 결과가 나온다. 다음은 **재현되지 않을 수 있다**:

* 다른 BLAS/스레드 설정에서의 부동소수점 축약 순서 (manifest 에 스레드 설정 기록)
* `recording.mode = "summary"` 로 실행한 경우의 **개별 사건 추적** (집계만 남는다)
* `recording.mode = "selected"` 에서 선택되지 않은 뉴런의 사건
* `poisson` 모드에서 한 스텝에 2건 이상 발생한 사건 (1 스파이크로 잘리며,
  잘린 건수는 `errors.jsonl` 에 경고로 기록된다)

---

## 12. 자동 실행 출력 (`run-all`)

`run-all --out <폴더>` 는 아래 구조를 만든다. 각 단계 폴더의 내부는 위에서
설명한 실행 기록 구조 그대로다.

```
<출력폴더>/
  summary.json      전체 요약 (아래)
  SUMMARY_ko.md     사람이 읽는 한국어 요약. summary.json 에서만 생성한다
  run_all.log       진행 로그
  <설정이름>/
    01_config_check/inspect.json
    02_validate/    03_simulate/    04_experiment/    05_reference/
```

번호는 고른 단계 순서가 아니라 **고정 위치**다. 일부 단계만 돌려도 전체 실행과
같은 경로가 나온다.

`summary.json`:

```
started_utc, finished_utc, elapsed_sec, command, output_dir,
configs[], stages[], dry_run, backend_choice, library_versions,
experiment_status, all_stages_ok,
results: {
  <설정이름>: {
    config, config_sha256, notes[],        notes 에 덮어쓴 설정 값이 남는다
    stages: [ {stage, title_ko, status, elapsed_sec, output_dir, detail, error} ]
  }
}
```

`status ∈ {completed, failed, skipped, dry_run}`. `skipped` 는 `completed` 에
포함하지 않는다.

`05_reference/reference.json` 에는 Rao 참조 모델 요약·유한차분 결과, 고정 Gabor
대조 경로의 방향 튜닝·위상 스윕·명암 반전, `explain_neuron` 결과, 소거 재실행
비교가 들어간다. 고정 Gabor 경로의 선택성은 **학습된 것이 아니다** (그 사실이
`learned: false` 로 함께 저장된다).
