# visual_cortex_gpu 4.0.17 — 메뉴 27 사용 순서 (입력 정보 보존·규모 재구성, capacity_rebuild)

프로그램 파일: `visual_cortex_gpu_4_0_17_capacity_rebuild.py`
(기준: 4.0.16 `visual_cortex_gpu_4_0_16_cosine_target_shapes.py`, SHA256 `a7c70986…934456`. 메뉴 0~26 유지, 메뉴 27 추가)

## 1. 무엇을 시험하는가

64 px 축소, 과도한 블러, 성긴 망막 표본, 입력 누락 때문에 도형 구별 정보가 약해졌을 수 있다는 **가설**을 시험한다.
블러가 유일한 원인이라고 확정하지 않는다. 임계값이나 그 뒤 전달 병목이 함께 있을 수 있다.

| 단계 | L4 / 입력 | L2 | L3 |
|---|---|---|---|
| Retina | 576 곳 × 9 채널 = 5,184 | | |
| LGN | 5,184 | | |
| V1 | 5,184 | 1,024 (32×32) | 1,024 |
| V2 | 1,024 | 256 (16×16) | 256 |
| V4 | 256 | 128 (8×16) | 128 |
| IT | 128 | 64 (8×8) | 64 (= IT 출력 차원) |

- 주 감각 슬롯 합: **19,904**.
  - 이 구성은 물리 억제성 세포와 L1/L5/L6 세포를 만들지 않는다. 그래서 전체 할당 뉴런 수도 같다 (A 의 S1 검사가 확인).
  - L1 버퍼·교정 패킷·복원기 텐서는 뉴런 수에 넣지 않는다.
- **일대일 연결 (k → k)**: Retina→LGN, LGN→V1 L4, 영역 안 L2→L3, L3→다음 영역 L4.
- **L4→L2 국소 수렴**:
  - V1 은 고정 Gabor 이다. ON 입력은 양의 엽, OFF 입력은 음의 엽에서 받고, 저주파 3채널은 따로 가우시안으로 결합한다.
  - V2/V4/IT 는 국소 가우시안에 방향 유사도 가중을 곱한다.
  - 쓰이지 않는 L4 는 반경을 단계적으로 늘리는 보완 절차로 배정하고, 그 내용을 기록한다.
- **수신 예산**: L2 마다 Σw = G (nS).
  - 기본 G 는 기존 15 mV 보정식을 단일 입력에 적용한 값이다. 라벨이나 정확도와는 무관하다.
- **임계값 (E_L 기준)**: L2/L3 3 mV, L4 5 mV. 억제 배율은 0.0 이다.
- **입력**: 원본 200 px 를 그대로 쓴다.
  - DoG/저주파 폭은 기존 1/3/6 px 와 같은 시야 각도 (0.1875/0.5625/1.125 도) 이다.
  - 실제 px 값도 함께 기록한다.
- 약 2만 개 구성은 공학적 출발안이다. 최적 크기, 뇌 모사, 학습 성공 중 어느 것도 보장하지 않는다.

## 2. 파일 배치와 실행

```
<PyCharm 프로젝트 폴더>/
├─ visual_cortex_gpu_4_0_17_capacity_rebuild.py
├─ shapes/      ← 1순위 (circle/ square/ star/ triangle/)
└─ archive.zip  ← 2순위
```

1. 인자 없이 실행한다.
2. 결과 저장 폴더를 입력한다.
3. preset 은 아무 값이나 넣는다 (메뉴 27 은 자기 구성을 쓴다).
4. 장치를 고른다 (auto / cuda / cpu).
5. 주 메뉴에서 **27** 을 고른다.

- 데이터 경로는 `D` 로 바꾼다 (메뉴 26 과 같은 탐색 규칙).
- 패키지는 numpy, scipy, torch, pillow 이고 matplotlib 은 그림용이다. 프로그램은 패키지를 자동 설치하지 않는다.
- GPU 메모리가 부족해도 뉴런 수, 해상도, 시간창, batch 를 **자동으로 줄이지 않는다**.
  - 기록을 남기고 멈춘다 (`run_status.json` 또는 `capacity_rebuild_failure.json`).
  - batch 는 `S` 에서 직접 줄인다. 기본값은 1 이다.

## 3. 메뉴 27 실행 순서

| 순서 | 메뉴 | 하는 일 | 신경망 실행 |
|---|---|---|---|
| ① | **27-1** A 구조 검사 | 구조를 CPU 에서 만들고 S1~S13 을 검사한다 (슬롯 수, 일대일 계약, L4→L2 사용 범위, 채널 의미, 예산, 역조회, 인덱스, 순방향 그래프, 지연, 방향 bank, 입력 사용 범위, 임계값 계획). 이론 메모리 추정도 낸다 | 없음 (시뮬레이션 없음) |
| ② | **27-2** B 전달 진단 | train 표본 클래스당 16 장을 학습 없이 통과시킨다. 단계별 발화, 흥분 도착, 무반응, 첫 발화 시각, 포화, 클래스 내/간 코사인, 최초 소실 단계를 기록한다 | 있음 (학습·교정 없음) |
| ③ | **27-3** 기존 구성 비교 | 같은 표본을 기존 hierarchy_small paired_only (64 px) 구성에도 통과시킨다. 해상도·배선·뉴런 수가 함께 바뀐 **복합 비교**다 | 있음 |
| ④ | **27-4** 전처리 분리 | F0~F5 에서 해상도, 표본 수, 사전 블러, DoG 폭을 하나씩 바꿔 입력 표현을 비교한다 | 없음 (입력 표현만) |
| ⑤ | **27-5** C 준비 | 완료된 B 폴더를 지정한다. 정규화, 복원기, 프로토타입, Q_0 (64 차원) 를 만든다. 클래스 평균이 0 인 클래스가 있으면 `invalid` 로 멈추고 학습하지 않는다 | 있음 |
| ⑥ | **27-6** D 학습 | 유효한 C 폴더를 지정한다. C 의 준비 상태와 정규화를 그대로 쓴다. frozen 과 local_threshold_correction 을 학습하고 비교한다 | 있음 |
| ⑦ | **27-9** 최종 test | 정식 (formal) D 의 확정 체크포인트로 평가한다. 접근 횟수가 누적 기록된다 | 있음 |
| ⑧ | **27-10** 추론 | D 체크포인트로 이미지 폴더나 파일을 추론한다 | 있음 |

- **27-7**: 중단된 D 재개 (같은 설정만).
- **27-8**: 저장 기록만으로 보고서와 그림을 다시 만든다. 기록 파일이 바뀌지 않았는지도 확인한다.
- **27-11**: 메뉴 26 의 회로 검사 C1~C7 을 새 구성 위에서 작은 표본으로 돌린다.
- **S**: 설정을 바꾼다.
- **J**: 설정 JSON 을 불러오거나 저장한다.
- **H**: 사용법을 보여 준다.

조건 이름: 기록에는 메뉴 26 의 이름이 그대로 남는다.
- `frozen` = `frozen_all`
- `local_threshold_correction` = `angular_hierarchical`

## 4. 결과 위치

실행마다 `출력 폴더/capacity_rebuild_<단계>_<시각>_<ID>/` 를 만든다.

| 파일 | 내용 |
|---|---|
| `REPORT.md` | 한국어 보고서. 저장 파일의 값만 옮긴다 |
| `resolved_config.json`, `source_sha256.txt`, `environment.json` | 실제 구성, 코드 해시, 장치·결정론 기록 |
| `neuron_table.csv`, `slot_map.npz`, `edges.npz`, `connectivity_checks.json` | 뉴런 표, 슬롯 맵, 전체 edge (정·역방향 CSR), 구조 검사 결과 |
| `dataset_manifest.csv`, `split_manifest.csv`, `diagnostic_sample_ids.json` | 데이터 목록, 분할, B 진단 표본 |
| `stage_metrics.csv`, `transmission_chain.csv`, `class_separability.json` | B 의 단계별 지표, 최초 소실 단계, 클래스 분리 설명 통계 |
| `frontend_features.npz`, `it_vectors.npz`, `neuron_activity_*.npz` | 입력 표현, IT 64 차원 출력, 뉴런별 활동 |
| `runtime_metrics.csv` | 이 실행의 실측 시간과 GPU peak 메모리 |
| `preparation_status.json`, `checkpoints/`, `epoch_metrics.csv`, `learning_section_ct.md` | C/D 의 준비 상태, Q 버전, 체크포인트, 에폭 기록 |
| `figures/` | 원본 vs 전처리, 표본 위치·RF, 단계별 활동 지도 (보간 없음), 첫 발화·발화 수·무반응, IT 출력·유사도 행렬, 전처리 분리 |

## 5. 명령줄

```
python visual_cortex_gpu_4_0_17_capacity_rebuild.py --mode capacity_rebuild --cr-task structure --output "D:\CortexResults"
python visual_cortex_gpu_4_0_17_capacity_rebuild.py --mode capacity_rebuild --cr-task transmission --device cuda --output "D:\CortexResults"
python visual_cortex_gpu_4_0_17_capacity_rebuild.py --mode capacity_rebuild --cr-task compare_legacy --device cuda --output "D:\CortexResults"
python visual_cortex_gpu_4_0_17_capacity_rebuild.py --mode capacity_rebuild --cr-task frontend_split --output "D:\CortexResults"
python visual_cortex_gpu_4_0_17_capacity_rebuild.py --mode capacity_rebuild --cr-task prepare --cr-b-run "<B 폴더>" --device cuda --output "D:\CortexResults"
python visual_cortex_gpu_4_0_17_capacity_rebuild.py --mode capacity_rebuild --cr-task train --cr-c-run "<C 폴더>" --cr-epochs 2 --device cuda --output "D:\CortexResults"
python visual_cortex_gpu_4_0_17_capacity_rebuild.py --mode capacity_rebuild --cr-task resume --run-dir "<D 폴더>" --output "D:\CortexResults"
python visual_cortex_gpu_4_0_17_capacity_rebuild.py --mode capacity_rebuild --cr-task report --run-dir "<아무 메뉴 27 폴더>"
python visual_cortex_gpu_4_0_17_capacity_rebuild.py --mode capacity_rebuild --cr-task final_test --run-dir "<정식 D 폴더>" --output "D:\CortexResults"
python visual_cortex_gpu_4_0_17_capacity_rebuild.py --mode capacity_rebuild --cr-task infer --run-dir "<D 폴더>" --cr-condition full_0_460__angular_hierarchical --cr-image "<이미지>" --output "D:\CortexResults"
```

- 공통 선택 옵션:
  - `--cr-settings 설정.json`
  - `--cr-data 경로`
  - `--cr-run-kind quick|formal`
  - `--cr-batch-size`
  - `--cr-diag-per-class`
  - `--cr-seed`
  - `--cr-nondeterministic`
- C 와 D 는 같은 run_kind 와 같은 표본 설정이어야 한다. 다를 수 있는 것은 에폭 수뿐이다.

## 6. 개발 단계 검사 상태

- 실행한 것은 정적 검사뿐이다.
  - 구문 컴파일
  - pyflakes: 새 경고 0. 기존 원본에 있던 경고 1건은 그대로다.
  - AST 기반 이름·속성·호출 인자 대조
- mypy 는 이 환경에 설치되어 있지 않아 실행하지 못했다.
- **NOT_RUN**:
  - 27-1~27-11 전체 (19,904 슬롯 실제 생성과 S1~S13 포함)
  - 메뉴 0~26 회귀 실행
  - GPU 메모리 실측, 실행 시간, CUDA 결정론 재현성
- 성능, 정확도, 정보 보존 개선 여부는 사용자가 실행한 결과 폴더의 기록으로만 판단한다.

## 7. 주의

- 기존 구성과의 비교 (27-3) 는 복합 변경이다. 원인을 하나로 확정하지 말고 27-4 와 함께 본다.
- IT 출력이 64 차원이라는 사실만으로 클래스 분리가 생겼다고 보지 않는다.
- 총 수신 가중치가 같아도 동시 입력 전류가 같다는 보장은 없다. 실제 도착 전도도는 B 에서 따로 기록한다.
- L1/L5/L6 교정 역할은 계산 가정이다. 실재 피질의 확정된 학습 알고리즘이 아니다.
- 외생 입력은 [batch, 입력] 프레임과 시간표로 저장한다.
  - 예외: 주의 (attention) 배율이 1 이 아닌 경로에서는 기존 연산이 [time, batch, 입력] 배열을 만든다. 메뉴 27 의 기본 실행 계약에는 이 경로가 없다.
