# visual_cortex_gpu 4.0.16 — 메뉴 26 사용 순서 (목표 벡터 코사인 판정 + 4종 도형 이미지)

프로그램 파일: `visual_cortex_gpu_4_0_16_cosine_target_shapes.py` (기준: 4.0.15 `visual_cortex_gpu_4_0_15_readout_window_compare.py`, 메뉴 0~25 유지, 메뉴 26 추가)

## 1. 파일 배치 (PyCharm 프로젝트 폴더)

```
<PyCharm 프로젝트 폴더>/
├─ visual_cortex_gpu_4_0_16_cosine_target_shapes.py
├─ shapes/                 ← 1순위 (있으면 이것을 쓴다)
│   ├─ circle/   *.png
│   ├─ square/   *.png
│   ├─ star/     *.png
│   └─ triangle/ *.png
└─ archive.zip             ← 2순위 (shapes/ 가 없을 때. 압축을 풀지 않아도 된다)
```

- 데이터는 **.py 파일이 있는 폴더 기준**으로 찾는다 (PyCharm 작업 폴더와 무관). 둘 다 없으면 메뉴 26 의 `D` 로 경로를 입력한다.
  - 받는 경로: `shapes` 폴더 자체 / `shapes` 를 포함한 상위 폴더 / `archive.zip`
  - 공백·한글·따옴표가 들어간 Windows 경로를 그대로 붙여 넣어도 된다.
- archive.zip 은 이미지 항목만 읽는다. 상위 경로 탈출, 절대 경로, 심볼릭 링크, 암호화 항목은 읽지 않는다. 동봉된 `.py` 같은 파일은 실행하지 않는다.
- 클래스 폴더 이외의 폴더는 무시하고 기록만 한다. 클래스별 이미지 수는 고정값으로 가정하지 않고 실제로 센다.
- 결과는 매 실행마다 `출력 폴더/cosine_target_shapes_<종류>_<UTC 시각>_<ID>/` 아래에 새로 만든다. 데이터 원본 폴더에는 쓰지 않는다.

## 2. 실행 환경

- 필요한 패키지: numpy, scipy, torch (CUDA 판), matplotlib, Pillow.
  - 프로그램은 패키지를 **자동 설치하거나 다시 설치하지 않는다**. 지금 잘 동작하는 torch 를 그대로 쓴다.
  - Pillow 가 없을 때만 직접 `pip install pillow` 한다.
- PyCharm 에서 이 .py 를 인자 없이 실행한다.
  1. 결과 저장 폴더를 입력한다.
  2. 주 메뉴에서 `26` 을 고른다.
- GPU 메모리가 부족하면 CPU 로 몰래 바꾸지 않고 멈춘 뒤 안내를 남긴다. 그때는 `S` 에서 batch 를 줄이고 `26-6` 으로 재개한다.
- 이미지 해독 작업 프로세스 수의 기본값은 0 이다 (Windows 안전값). 1 이상은 `S` 에서 바꾼다.

## 3. 메뉴 26 실행 순서

| 순서 | 메뉴 | 하는 일 | 신경망 실행 |
|---|---|---|---|
| ① | **26-1** 데이터 점검 | 선택된 경로, 클래스별 이미지 수, 읽기 실패, 크기·모드, 분할 결과, 중복·라벨 충돌, 미리보기 `preview_model_input.png` | 없음 |
| ② | **26-2** 검사 A/B | B1~B10 코사인 판정 규칙, A1~A5 데이터·계약 검사 | 없음 |
| ③ | **26-3** 회로 검사 C | C1~C7. 작은 표본으로 준비 1회, 에피소드 몇 개, 1 에폭 | **있음** (물리 엔진) |
| ④ | **26-4** 빠른 점검 학습 | 클래스당 train 8 / dev 4, 2 에폭, batch 8. 코드 경로 확인용이며 정식 실험이 아니다 | 있음 |
| ⑤ | **26-5** 정식 train/dev 학습 | split 뒤 train·dev 전체, 10 에폭, batch 10. 시작 전 계획을 보여 주고 묻는다 | 있음 |
| ⑥ | **26-9** 추론 | 학습 폴더의 추론 체크포인트 (Θ, Q_eval) 로 이미지 한 장 또는 폴더를 판정한다 (라벨 폴더가 없어도 된다) | 있음 |
| ⑦ | **26-8** 최종 test | 설정을 확정한 **정식** 학습 폴더에서 test 를 평가한다 | 있음 (test) |

- 중단된 실행은 **26-6** 으로 재개한다. 원 설정과 데이터 fingerprint 를 그대로 쓴다.
- **26-7** 은 저장된 기록만으로 report.md 와 그림을 다시 만든다 (신경망 실행 없음).
- `D` 데이터 경로 / `O` 결과 폴더 / `S` 설정 / `H` 도움말 / `0` 이전 메뉴.

### 단계별 확인 사항

- **26-1**
  - `split_summary.json`: train/dev/test 개수와 제외 사유 (gap, 중복, 라벨 충돌, 읽기 실패).
  - `class_to_idx.json`: circle=0, square=1, star=2, triangle=3.
  - `dataset_manifest.csv`, `preview_model_input.png`: 모델 입력 모양과 흑백 극성.
- **26-2**: `checks.json` 의 각 항목이 PASSED 인지 본다. FAILED 가 있으면 다음 단계로 가지 않는다.
- **26-3**: C1~C7 결과를 본다 (teacher=0 무수정, reset 뒤 theta_base 보존, 평가 순수성 등).
- **26-4**: `report.md` 와 `epoch_metrics.csv` 가 만들어지는지, 계획 화면의 개수가 맞는지 확인한다. 수치는 해석하지 않는다.
- **26-5**: 끝나면 `report.md`, `run_record.md`, `epoch_metrics.csv`, `per_sample_metrics.csv`, `correction_metrics.csv`, `confusion_matrix.csv`, `target_bank.json/.npz`, `access_counters.json` (test 신경망 평가 0), `checkpoints/<조건>/inference_checkpoint.*` 를 본다.
- **26-9**: 출력은 예측 클래스, 네 목표와의 코사인 유사도, top1 − top2, ‖h‖, 상태 (OK / NO_RESPONSE / AMBIGUOUS) 다. 전체는 `inference_results.csv` 에 있다.
- **26-8**
  - 정식 학습 (`cosine_target_shapes_formal_...`) 이 완료된 폴더만 받는다.
  - 다시 평가하면 횟수와 체크포인트 hash 가 누적 기록되며, 독립 시험이 아니다.
  - dev/test 결과를 보고 설정을 고친 뒤 다시 test 하는 반복에 쓰지 않는다.

## 4. 판정 규칙과 조건

- 최종 판정
  - IT L3 출력 h 와 네 목표 방향 Q (각 클래스 하나) 의 코사인 유사도로 판정한다. 가장 큰 값이 예측이다. 분류기 C 는 학습하지도 쓰지도 않는다.
  - ‖h‖ ≤ eps 이면 `NO_RESPONSE`, 상위 두 값이 허용오차 안이면 `AMBIGUOUS` 다. 둘 다 오답으로 정확도 분모에 포함된다.
- 코사인 값은 유사도이지 확률이 아니다. 균등 무작위 기대 정확도는 25% 이며, 이전 7 클래스 결과와 비교하지 않는다.
- 기본 판독 창은 full_0_460 하나다. 두 조건을 돌린다.
  - `full_0_460__frozen_all`: 임계값을 학습하지 않는다 (단계 A 결과 재사용, Q_0 판정).
  - `full_0_460__angular_hierarchical`: 계층적 임계값 교정을 한다. 에폭 e 동안 Q_e 를 고정한다. 다음 에폭의 Q_next 는 train 으로만 계산한다.
- 체크포인트는 마지막 에폭 것을 쓴다 (dev 로 고르지 않는다). 추론 체크포인트에는 Θ 와 그 에폭 평가에 쓴 Q_eval 을 함께 저장한다.
- 분할 규칙 `temporal_block_with_gap`
  - 클래스마다 파일 이름을 자연 숫자 순서로 정렬한 뒤 70/15/15 로 나눈다.
  - 각 경계 양쪽 50 개는 제외한다.
  - SHA256 과 화소 hash 가 같은 항목이 다른 분할에 걸치면 그 묶음을 모두 제외한다.
- 입력 정규화 계수와 준비 (D/T/scale, Q_0) 는 train 에서만 정한다. 증강은 없다. 흑백 반전은 기본 OFF 이며, `S` 에서 전체에 일괄 적용하는 것만 가능하다.

## 5. 명령줄로 실행할 때 (선택)

```
python visual_cortex_gpu_4_0_16_cosine_target_shapes.py --mode cosine_target_shapes --ct-task data_check --output 결과폴더
python visual_cortex_gpu_4_0_16_cosine_target_shapes.py --mode cosine_target_shapes --ct-task pure_check --output 결과폴더
python visual_cortex_gpu_4_0_16_cosine_target_shapes.py --mode cosine_target_shapes --ct-task check      --output 결과폴더
python visual_cortex_gpu_4_0_16_cosine_target_shapes.py --mode cosine_target_shapes --ct-task quick      --output 결과폴더
python visual_cortex_gpu_4_0_16_cosine_target_shapes.py --mode cosine_target_shapes --ct-task formal     --output 결과폴더
python visual_cortex_gpu_4_0_16_cosine_target_shapes.py --mode cosine_target_shapes --ct-task infer --output 결과폴더 --run-dir 정식학습폴더 --ct-condition full_0_460__angular_hierarchical --ct-image 이미지_또는_폴더
python visual_cortex_gpu_4_0_16_cosine_target_shapes.py --mode cosine_target_shapes --ct-task final_test --output 결과폴더 --run-dir 정식학습폴더
```

- 데이터 경로를 직접 줄 때는 `--ct-data "경로"` 를 쓴다.
- 재개는 `--ct-task resume --run-dir 실행폴더`, 보고서 재생성은 `--ct-task report --run-dir 실행폴더` 다.
- 그 밖의 옵션: `--ct-epochs`, `--ct-batch-size`, `--ct-train-per-class`, `--ct-dev-per-class`, `--ct-gap`, `--ct-invert`, `--ct-three-windows`, `--ct-diagnostics`, `--ct-num-workers`, `--ct-checkpoint-every`.

## 6. 이 파일을 만든 환경에서 한 것과 하지 않은 것

- **한 것**
  - 소스 정적 점검: `py_compile`, pyflakes (4.0.15 부터 있던 경고 1건 외 없음).
  - 메뉴·CLI 구조 점검.
  - 가짜 엔진 (NumPy 대역, 물리 아님) 과 작은 합성 이미지 묶음으로 코드 경로 확인: 데이터 읽기, 분할, 학습 흐름, 재개, 최종 test 등록, 추론, 보고서. 이 결과는 정확도나 실험 결과가 아니다.
  - 이전 메뉴 회귀 스크립트 재실행.
- **하지 않은 것 (NOT_RUN)**
  - 검사 A1~A5, B1~B10, C1~C7. 프로그램 안에 넣어 두었고, 사용자가 26-2 / 26-3 에서 실행한다.
  - 실제 물리 시뮬레이션, 학습, smoke 실행, 파라미터 탐색, 분류기 학습. 정확도 예측값도 만들지 않았다.
  - 실제 archive.zip / shapes 데이터 검사. 이번 요청에 데이터 파일이 첨부되지 않았으므로 26-1 에서 확인한다.
