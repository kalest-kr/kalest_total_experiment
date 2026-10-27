# 입력 크기 기반 희소(근사) 역전파 비교 실험

MNIST MLP(784 → 256 → 128 → 10, ReLU, 편향 사용)에서 순전파는 그대로 두고, 역전파 때 각 수신 노드의
상위 1개 또는 2개 입력 연결에만 기울기를 전달하면 정확도, 학습 시간, 메모리가 어떻게 달라지는지 비교합니다.
가중합 기울기의 일부를 일부러 빼므로 이 방식은 **근사 역전파**이며, 성능이 좋아진다고 가정하지 않습니다.

> **검증 상태: 코드 작성 및 정적 검토 완료, 실행 검증 미실시.**
> 작성자는 데이터 다운로드, 학습, 평가, 벤치마크, 자체 검증을 **실행하지 않았습니다**.
> 이 문서에는 실행 결과나 성능 수치가 없습니다. 모든 수치는 사용자가 직접 실행해서 얻어야 합니다.

## 1. 설치 (Windows)

Python 3.10 이상(64비트)이 필요합니다. 명령 프롬프트(cmd)에서 이 폴더로 이동한 뒤:

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

PowerShell에서 `activate`가 막히면 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`를 한 번 실행하거나 cmd를 쓰세요.

**CPU만 쓸 때**

```bat
pip install -r requirements.txt
```

**NVIDIA GPU(CUDA)를 쓸 때**: torch/torchvision을 PyTorch 공식 색인에서 **먼저** 설치한 뒤 나머지를 설치합니다.
`cu126` 부분은 예시입니다. https://pytorch.org/get-started/locally/ 에서 본인 드라이버에 맞는 명령을 확인하세요.

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

마지막 줄이 `True`를 출력하면 CUDA를 사용할 수 있습니다.

## 2. CPU / CUDA 선택

```bat
REM 기본값 auto: CUDA가 있으면 CUDA, 없으면 CPU
python sparse_backprop_experiment.py --run-all --device auto
python sparse_backprop_experiment.py --run-all --device cpu
REM cuda: CUDA를 쓸 수 없으면 오류로 멈춤
python sparse_backprop_experiment.py --run-all --device cuda
```

## 3. 전체 실행

```bat
python sparse_backprop_experiment.py --run-all
```

순서: 환경 확인 → 데이터 준비(없으면 MNIST 다운로드) → 자체 검증 → 5개 조건 × 시드 3개(0, 1, 2) 학습 →
조건별 최종 시험 평가 → 요약표·그래프 생성. 진행 상황은 한국어로 출력되며 중간에 입력을 묻지 않습니다.

기본값: 10 에폭, 배치 128, SGD 학습률 0.05, momentum 0, weight_decay 0, 분할 시드 2024(학습 50,000 / 검증 10,000),
적용 범위 `all`(최종 분류층 포함). 모든 값은 옵션으로 바꿀 수 있습니다(`python sparse_backprop_experiment.py -h`).
주요 옵션: `--epochs --batch-size --lr --seeds --data-dir --output-dir --sparse-layers all|hidden
--selection-chunk-size --target-accuracy --conditions --selection-stats on|off --deterministic`.

자체 검증이 하나라도 실패하면 본학습을 시작하지 않고 실패한 검사, 원인, 소스 위치(파일:줄)를 출력합니다.

## 4. 짧은 확인 실행

```bat
REM 자체 검증만 (데이터 다운로드 없음)
python sparse_backprop_experiment.py --self-test-only

REM 1 에폭, 시드 1개, 에폭당 50 배치만, 세부 측정 축소
python sparse_backprop_experiment.py --run-all --epochs 1 --seeds 0 --limit-train-batches 50 --profile-warmup 2 --profile-batches 5 --output-dir runs_quick

REM 검증 정확도 목표 도달 시간 기록 (0.97 또는 97 형식, 여러 개 가능)
python sparse_backprop_experiment.py --run-all --target-accuracy 0.97 0.98
```

## 5. 중단 후 재개

```bat
python sparse_backprop_experiment.py --resume runs_sparse_backprop\run_20261027_153000
```

- 완료된 조건(`summary.json` 있음)은 건너뛰고, 미완료 조건은 마지막으로 완료된 에폭 다음부터 이어갑니다.
  중단 시점에 진행 중이던 에폭은 처음부터 다시 학습합니다.
- 초기 가중치(`init/`), 학습/검증 분할(`split_indices.pt`), 옵티마이저·난수 상태(`last.pt`)를 저장하며,
  배치 순서는 (시드, 에폭)으로만 정해지므로 재개해도 같은 순서를 씁니다.
- 옵션을 생략하면 저장된 설정을 그대로 씁니다. 결과에 영향을 주는 설정(에폭, 학습률, 시드, 적용 범위, 장치 종류 등)을
  다르게 지정하면 결과를 섞지 않도록 **오류를 내고 멈춥니다**. `--target-accuracy`는 바꿔도 되며 요약만 다시 만듭니다.
- 실패한 조건은 `errors.log`와 `progress.json`에 기록되고 나머지 조건은 계속 진행됩니다. 재개하면 실패한 조건을 다시 시도합니다.

## 6. 결과 파일

```
runs_sparse_backprop\run_YYYYMMDD_HHMMSS\
  config.json                 실행 설정
  environment*.json           소프트웨어·하드웨어 환경, 비결정성 메모 (재개할 때마다 추가)
  progress.json               조건·시드별 상태 (pending / training / trained / completed / failed / interrupted)
  self_test_report*.json      자체 검증 결과
  run_log.txt, errors.log     진행 기록, 오류 상세
  init\seed_N.pt              시드별 공통 초기 가중치
  seed_N\<조건>\
    epochs.csv                에폭별 학습·검증 손실/정확도, 학습 시간, 검증 시간, 누적 학습 시간, 메모리
    profile.json              대표 배치 세부 시간(워밍업·측정 배치 수, 단계별 ms, 측정 추가 비용)
    summary.json              최고 검증 에폭, 시험 정확도/손실, 시간, 처리량, 메모리, 연결 선택 통계
    selection_counts.npz      층별 연결 선택 횟수 [출력, 입력]
    last.pt, best.pt          최신 체크포인트, 검증 최고 체크포인트
  summary\
    runs_summary.csv          조건 × 시드 요약
    condition_mean_std.csv    조건별 평균(_mean)과 표본 표준편차(_std)
    target_accuracy.csv       (목표값을 준 경우) 처음 도달한 에폭·누적 학습 시간, 못 하면 "미도달"
    fig_val_acc_vs_epoch.png  검증 정확도 대 에폭
    fig_val_acc_vs_time.png   검증 정확도 대 누적 학습 시간
    fig_test_acc.png          시험 정확도 비교
    fig_time_memory.png       학습 시간, 단계별 시간, 메모리 비교
    run_config_and_environment.json
```

읽는 법:

- 정확도(`*_acc`)는 0~1 비율입니다(그래프는 %). 시험 정확도는 **검증 정확도가 가장 높았던 에폭**(동률이면 이른 에폭)의
  체크포인트로 학습이 끝난 뒤 한 번만 평가합니다. 시험 결과로 모델이나 설정을 고르지 않습니다.
- `total_train_time_s`: 학습 루프 벽시계 시간의 합(검증, 체크포인트 저장, 세부 측정 제외).
  `train_time_ratio_vs_dense`: 같은 시드 dense 대비 **실측** 학습 시간 비율(1보다 작으면 빠름).
- `backward_terms_ratio_theory`: 역전파 곱셈 항 수의 **이론** 비율(선택한 항의 수 감소). 실측 속도와 다릅니다.
- `prof_*_ms`: 학습과 분리된 초기 가중치 사본으로 잰 대표 배치의 단계별 시간입니다. 단계마다 장치를 동기화하므로
  그만큼 느려지며, 그 추가 비용은 `prof_sync_overhead_ms`(동기화 측정 합 − 동기화 없는 측정)로 따로 기록합니다.
  `prof_forward_incl_selection_ms`는 선택 인덱스 계산을 포함한 순전파, `prof_selection_only_ms`는 선택 계산만입니다.
- 메모리: CUDA는 학습 루프의 `max_memory_allocated`와 그 기준값 대비 증가분(`cuda_train_peak_minus_baseline_mb`).
  CPU는 psutil 프로세스 RSS를 25 배치마다 표본 추출한 값이라 짧은 순간의 최대값을 놓칠 수 있고, 인터프리터·라이브러리·
  할당자 캐시를 포함한 프로세스 전체 값입니다(텐서 메모리만의 값이 아님). `saved_tensor_mb`는 역전파용으로 그래프가
  붙잡는 텐서의 정확한 크기(파라미터·입력 배치 제외)입니다.
- `never_selected_fraction_*`: 학습 동안 한 번도 선택되지 않은 연결 비율. 이 연결의 가중치는 초기값에서 바뀌지 않습니다.
- 기본 실행은 선택 빈도 집계(`--selection-stats on`) 비용이 근사 조건의 학습 시간에 포함됩니다(배치당 비용은
  `prof_selection_stats_ms`). 순수 시간 비교가 목적이면 `--selection-stats off`로 따로 실행하세요.
- CUDA의 원자적 누적(scatter_add_, bincount)과 cuBLAS는 같은 시드라도 결과가 비트 단위로 같지 않을 수 있습니다.
  `environment.json`의 `nondeterminism_notes`에 기록됩니다.

## 7. 다섯 조건과 근사 역전파 수식

배치 입력 `X[b,i]`, 가중치 `W[j,i]`, 역전파로 도착한 값 `Delta[b,j]`라고 할 때:

```
순전파 (모든 조건 동일)   Z[b,j] = Σ_i W[j,i] X[b,i] + bias[j]   (은닉층 뒤 ReLU)
선택 점수   input        : s[b,j,i] = |X[b,i]|
            contribution : s[b,j,i] = |W[j,i] X[b,i]|
선택 마스크 M[b,j,i] = 1[i 가 (b, j)의 점수 상위 k 안에 있음]
근사 dW[j,i] = Σ_b M[b,j,i] Delta[b,j] X[b,i]
근사 dX[b,i] = Σ_j M[b,j,i] Delta[b,j] W[j,i]
dbias[j]     = Σ_b Delta[b,j]           (일반 방식)
```

| 조건 | 역전파 |
|---|---|
| `dense` | 표준 `nn.Linear` 역전파 (기준군) |
| `input_top1` / `input_top2` | 원입력 \|x_i\| 상위 1/2개 연결. 같은 표본·같은 층의 모든 수신 노드가 같은 입력을 고름 |
| `contribution_top1` / `contribution_top2` | 가중 기여량 \|W_ji x_i\| 상위 1/2개 연결. 수신 노드마다 다른 입력을 고름 |

- 선택은 현재 순전파의 입력과 가중치로 하고, 그 결과를 해당 역전파에 그대로 씁니다. dW와 dX에 같은 마스크를 씁니다.
- 동률은 작은 입력 인덱스 우선(0 점수 포함), top-2는 서로 다른 두 인덱스, k가 입력 차원보다 크면 입력 차원까지만 선택합니다.
- 점수만 절댓값이고 실제 기울기 계산은 원래 부호를 씁니다. 손실의 배치 평균은 한 번만 적용하며, fan-in/k 같은 확대나
  선택되지 않은 항을 선택된 항에 더하는 보정은 하지 않습니다. ReLU와 CrossEntropy의 미분은 정상 계산합니다.
- 기본 적용 범위는 최종 분류층까지 모든 선형층이고, `--sparse-layers hidden`이면 은닉 선형층 두 개만 근사합니다.
- 평가는 선택 계산 없이 일반 밀집 MLP 순전파로 합니다.

## 8. 실제 가속을 보장할 수 없는 이유

구현은 선택된 항만 모아(`gather`, `index_select`) 계산하고 `scatter_add_`로 누적하며, 밀집 dW·dX를 만든 뒤
마스크를 곱하지 않습니다. 그래도 다음 비용이 남기 때문에 항 수가 줄어든 만큼 빨라진다고 볼 수 없습니다.

- **선택 비용**: input은 표본당 입력 수만큼 |x|와 argmax가 필요합니다. contribution은 표본당 `출력 × 입력` 개의
  |W x|를 계산해야 하므로, 아끼려는 밀집 dW 계산과 같은 규모의 연산이 순전파에 추가됩니다.
- **저장 비용**: 선택 인덱스(int64)와 선택된 입력값을 역전파까지 보관합니다. contribution은 `출력 × 배치 × k` 크기입니다.
  ReLU는 여전히 자기 출력을 저장하므로 활성값 메모리가 크게 줄지 않습니다.
- **밀집으로 남는 연산**: 순전파 행렬곱, 가중치 기울기 버퍼의 0 초기화(`출력 × 입력`), 편향 기울기, 밀집 가중치 전체에 대한
  SGD 갱신, 평가 순전파는 모든 조건에서 그대로입니다.
- **하드웨어 특성**: 밀집 행렬곱은 cuBLAS/MKL로 매우 최적화되어 있고, 작은 MLP에서는 GPU가 커널 실행 오버헤드에 묶이기
  쉽습니다. 불규칙한 gather/scatter와 추가 커널 호출이 오히려 느릴 수 있으며 CPU와 GPU의 결과도 다를 수 있습니다.

그래서 결과에서는 이론적 항 수 비율(`backward_terms_ratio_theory`)과 실측 시간 비율(`train_time_ratio_vs_dense`,
`prof_backward_ms`)을 따로 보고합니다.

## 9. 작성자 실행 여부

이 프로그램과 자체 검증 함수는 작성만 했고 작성자는 **어떤 것도 실행하지 않았습니다**(데이터 다운로드, 학습, 평가,
벤치마크, 자체 검증 모두 미실시). 정적 검토로 문법 확인(AST 파싱)과 정의되지 않은 이름 검사만 했습니다.
첫 실행은 `--self-test-only` 또는 4절의 짧은 확인 실행으로 시작하길 권합니다.
