# 3×3 뉴런 기록 구조 시각피질 시뮬레이터 (연구용)

망막 – LGN – V1 – V2 – V3 – V4 – IT 의 **공간 배치, 층별 연결, 시간에 따른 신호
전달, 국소 학습**을 관찰할 수 있는 Python 시뮬레이터다. 핵심은
"어느 입력이 언제 도착했고, 어떤 상태에서 발화했으며, 어디로 전달되었는가"를
**재현 가능하게 추적**하는 것이다.

> **이번 납품의 수치 실험 상태는 전부 `not_run` 이다.**
> 이 저장소를 만드는 동안 학습·시뮬레이션·데이터 생성·수치 검증·그래프 생성을
> 한 번도 실행하지 않았다. 실행은 **사용자가 명령을 내릴 때만** 일어난다.
> 자세한 내용은 `IMPLEMENTATION_STATUS.md` 참조.

이 프로젝트는 영역 이름만 붙인 일반 MLP 가 아니다. 공간 좌표, 세포 유형,
표적 층·구획, 흥분/억제, 재귀 연결, 전달 지연이 **실제 계산을 바꾼다**.
동시에 이 결과를 "인간 뇌의 완전하고 정확한 복제" 라고 부르지 않는다
(`BIOLOGY_AND_ASSUMPTIONS.md`).

---

## 1. 설치 (Windows PowerShell 기준)

```powershell
cd C:\경로\visual_cortex_3x3
python -m pip install -r requirements.txt
```

* Python **3.11 이상**이 필요하다. `python --version` 으로 확인하라.
* 필수 패키지: `numpy`, `scipy`, `Pillow`, `matplotlib`, `h5py`.
* **GPU 나 PyTorch 는 필요 없다.** 모든 필수 기능이 NumPy/SciPy 만으로 동작한다.
* 개발용(테스트) 패키지까지 설치하려면:

```powershell
python -m pip install -r requirements-dev.txt
```

* 가상환경을 쓰는 것을 권한다:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

* PowerShell 실행 정책 때문에 `Activate.ps1` 이 막히면:
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` 후 다시 시도하라.

### h5py 가 설치되지 않는 경우

대용량 기록(`events.h5`, `states.h5`, `updates.h5`)은 h5py 를 쓴다. 설치가 어렵다면
설정 파일에서 다음을 바꿔라.

```json
"recording": { "backend": "npz" }
```

npz 백엔드는 **메모리에 모았다가 종료 시 저장**하므로 대규모 실행에는 맞지 않는다.
이 한계는 manifest 에 기록된다.

---

## 2. 가장 쉬운 실행: `python run.py`

```powershell
python run.py
```

한국어 메뉴가 뜬다. **번호를 고르기 전에는 아무 것도 실행되지 않는다.**

```
  1) 최소 모델 검증 실행          (configs/minimal.json)
  2) V1 시뮬레이션 실행            (configs/v1_small.json)
  3) 전체 시각 경로 작은 모델 실험 (configs/hierarchy_small.json)
  4) 기존 실행 기록에서 보고서·그래프 생성
  5) 특정 뉴런의 입력·발화·출력 로그 조회
  6) 중단된 실험 재개
  7) 설정만 확인 (규모 추정, 실행하지 않음)
  8) 실행 기록 목록 보기
  0) 종료
```

* 각 메뉴에서 **설정 파일 경로와 출력 폴더**를 직접 지정할 수 있다 (엔터를 치면
  기본값을 쓴다). 경로에 **공백이나 한글이 있어도 된다.**
* 실행 중에는 진행 상황이 표시되고, **Ctrl+C** 로 언제든 중단할 수 있다.
  중단해도 기록과 체크포인트는 `runs/<실행ID>/` 에 남고 메뉴 6) 으로 재개할 수 있다.
* 0) 을 고르면 종료한다. **다른 실험을 자동으로 이어서 실행하지 않는다.**

### PyCharm 에서 실행하기

1. `File → Settings → Project → Python Interpreter` 에서 위에서 만든 가상환경
   (`.venv`) 을 고른다.
2. 프로젝트 루트의 `run.py` 를 열고 오른쪽 클릭 → `Run 'run'`.
3. 실행 구성의 *Working directory* 는 아무 곳이어도 된다. 기본 경로는
   `Path(__file__).resolve()` 로 해석하므로 다른 작업 디렉터리에서도 동작한다.
4. PyCharm 콘솔에서 입력을 받으려면 실행 구성의 **Emulate terminal in output
   console** 을 켜라.

---

## 2-1. 단일 파일 판 (`cortex_all_in_one.py`)

패키지 전체(`cortex/` 22개 모듈 + `run.py`)를 **파이썬 파일 하나**로 합친 판도 있다.
설정 4종을 코드로 내장해서 이 파일 하나만 있으면 동작한다.

```powershell
python cortex_all_in_one.py                                   # 한국어 메뉴
python cortex_all_in_one.py selftest                          # 내장 설정 자체 점검
python cortex_all_in_one.py inspect-config --config minimal
python cortex_all_in_one.py validate --config minimal --execute
python cortex_all_in_one.py simulate --config v1_small --execute
python cortex_all_in_one.py report --run-dir runs/RUN_ID
```

`--config` 에는 **내장 설정 이름**(`minimal`, `v1_small`, `hierarchy_small`,
`megapixel_input`) 또는 JSON 파일 경로를 줄 수 있다.

### 패키지 판과의 차이 (전부 파일 머리말에도 적혀 있다)

1. 상대 import 를 모두 없앴다. 모든 이름이 한 모듈 안에 있다.
2. 이름이 겹치던 최상위 함수의 이름을 바꿨다:

   | 원래 | 단일 파일 | | 원래 | 단일 파일 |
   |---|---|---|---|---|
   | `anatomy.build` | `build_anatomy` | | `rng.from_config` | `rng_from_config` |
   | `areas.build` | `build_wiring` | | `stimuli.generate` | `generate_stimuli` |
   | `plasticity.make` | `make_plasticity` | | `config.load` | `load_config` |
   | `predictive_coding.make` | `make_rao_model` | | `config.resolve` | `resolve_config` |
   | `cli.main` | `cli_main` | | `config.validate` | `validate_config` |

3. matplotlib 을 `_plt()` 로 **지연 로드**한다. 이 파일을 import 만 해도 백엔드
   초기화가 일어나지 않는다 (import 부작용 금지 규칙 유지).
4. 설정 4종을 `BUILTIN_CONFIGS` 로 내장했다.
5. `code_hash` 가 디렉터리 대신 이 파일 하나를 해싱한다.
6. 검증 14(import 부작용)가 이 파일을 경로로 import 하는 방식으로 바뀌었다.

계산 로직, 자료구조, 검증 기준, 단위 규약은 **패키지 판과 같다.**

### 두 판이 같은지 확인하는 방법

```powershell
python cortex_all_in_one.py selftest
```

내장 설정 4종을 해석해 `configs/*.json` 과 sha256 을 비교한다. 시뮬레이션은
실행하지 않는다.

### 단일 파일 다시 만들기

패키지 쪽을 고쳤다면 아래로 다시 생성한다 (이 스크립트도 실험을 실행하지 않는다).

```powershell
python tools/build_single_file.py
```

## 3. 반복 작업용 CLI

메뉴와 CLI 는 **같은 Runner/Recorder** 를 호출한다. 시뮬레이션 구현이 두 벌
있지 않다.

```powershell
python -m cortex.cli inspect-config --config configs/v1_small.json
python -m cortex.cli validate       --config configs/minimal.json --execute
python -m cortex.cli simulate       --config configs/v1_small.json --execute
python -m cortex.cli experiment     --config configs/hierarchy_small.json --execute
python -m cortex.cli resume         --run-dir runs/RUN_ID --execute
python -m cortex.cli report         --run-dir runs/RUN_ID
python -m cortex.cli explain-neuron --run-dir runs/RUN_ID --neuron-id 12 --from-ms 0 --to-ms 100
python -m cortex.cli figures        --run-dir runs/RUN_ID --neuron-ids 12,34 --rebuild-model
python -m cortex.cli list-runs
```

| 명령 | 기본 동작 | `--execute` |
|---|---|---|
| `inspect-config` | 설정과 규모만 출력 | (해당 없음) |
| `validate` | dry-run (규모만) | 필수 검증 1~14 실행 |
| `simulate` | dry-run | 자극 제시 시뮬레이션 실행 |
| `experiment` | dry-run | train/dev/test 분할 실험 실행 |
| `resume` | 재개할 체크포인트만 표시 | 실제 재개 |
| `report` / `explain-neuron` / `figures` / `list-runs` | **기존 기록만 읽는다** | (해당 없음) |

* **수치 실험 명령은 `--execute` 없이는 아무 것도 실행하지 않는다.**
* `report` 와 조회 명령은 기록이 없으면 "없다" 고 알려주고, **새 실험을 자동으로
  시작하지 않는다.**
* `validate` 의 구조적 필수 검사가 실패하면 의존 실험은 중지되고 실패 상태가
  기록된다.

---

## 4. 규모별 설정

| 설정 | 목적 | 엔진 모드 | 기록 모드 |
|---|---|---|---|
| `configs/minimal.json` | 소수 뉴런의 단일 전달·억제·지연·분기 확인 | `sum_threshold` | `full` |
| `configs/v1_small.json` | 작은 영상 + V1 미세회로, 방향·위상·주변 문맥 | `conductance_lif` | `selected` |
| `configs/hierarchy_small.json` | LGN + V1/V2/V3/V4/IT 작은 연결망 | `conductance_lif` | `selected` |
| `configs/megapixel_input.json` | 1024×1024 입력과 제한된 피질 표본 수 | `conductance_lif` | `summary` |

실행 전에 규모를 먼저 보라:

```powershell
python -m cortex.cli inspect-config --config configs/hierarchy_small.json
```

뉴런 수, 시냅스 수, 스텝 수, 이벤트 수, RAM/디스크 추정이 나온다.
**실제 런타임(초)은 측정 전에 확정하지 않으므로 `null` 로 나온다.**

용량 한도(`experiment.limits`, `wiring.max_total_synapses`,
`recording.max_events`)에 도달하면 **조용히 자르지 않고 명시적으로 중단**하고
체크포인트를 남긴다.

---

## 5. 실행 결과가 저장되는 곳

```
runs/<실행ID>/
  manifest.json      설정·해시·환경·단위·기록 모드·status
  events.h5          입력/도착/발화/학습 사건
  states.h5          선택 뉴런의 상태와 층별 집계
  updates.h5         가중치·임계 변화 (규칙별 항, clipping 전후)
  metrics.jsonl      실제 실행된 표본/스텝의 측정치와 측정 조건
  summary.csv        metrics 의 표 형태
  run.log            진행 로그
  errors.jsonl       경고·예외·중단 원인·재개 정보
  checkpoints/       가중치·상태·흔적·이벤트 큐·RNG 상태
  figures/           그림
  report.md          저장된 기록에서만 만든 한국어 보고서
```

`status` 는 `running / completed / interrupted / failed` 중 하나다.
**이미 `completed` 인 폴더는 덮어쓰지 않는다.** 다른 `--run-dir` 를 주어야 한다.

스키마 전체는 `DATA_SCHEMA.md` 에 있다.

---

## 6. 뉴런 하나를 들여다보기

```powershell
python -m cortex.cli explain-neuron --run-dir runs/RUN_ID --neuron-id 12 --from-ms 0 --to-ms 100
```

* 그 시간 구간에 **어떤 연결로 어떤 값이 언제 도착했는지**, 그때 적용된
  가중치·이득 스냅샷이 무엇인지, 발화가 언제 일어났는지를 돌려준다.
* 이것은 **관측된 기여 분석**이다. "이 연결이 발화를 일으켰다" 는 인과 주장을
  하려면 동일 상태·입력에서 그 연결을 제거한 **재실행 비교**가 필요하다
  (`cortex.analysis.ablation_rerun`).

3×3 기록 인터페이스를 직접 보려면:

```python
from cortex.config import load
from cortex import rng as rng_mod
from cortex.runner import build_model

cfg = load("configs/minimal.json")
model = build_model(cfg, rng_mod.from_config(cfg))
rec = model.anat.population.record(0)
m = rec.as_matrix()            # (3,3) object 배열
print(rec.as_matrix_description())
print(m[1, 0])                 # 출력 연결 목록 (입력 총합이 아니다)
print(m[2, 2].to_dict())       # 메타데이터
```

---

## 7. 중단과 재개

* 실행 중 **Ctrl+C** 를 누르면 안전하게 flush 하고 `status="interrupted"` 로
  기록한 뒤, 가능한 체크포인트를 저장한다.
* 재개:

```powershell
python -m cortex.cli resume --run-dir runs/RUN_ID --execute
```

* 재개할 때 **코드 해시와 설정 해시를 비교**한다. 다르면 오류를 낸다.
  차이를 알고도 진행하려면 `--allow-mismatch` 를 붙여라.
* `event_id` 카운터와 이벤트 로그 위치를 복원하므로 **사건이 중복 기록되지 않는다.**

---

## 8. 테스트

이 저장소를 만드는 동안 테스트를 실행하지 않았다. 사용자가 직접 실행한다:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`tests/` 의 pytest 는 단위 수준 검사이고, 명세의 **필수 검증 1~14** 는
`python -m cortex.cli validate --config <설정> --execute` 가 실행한다.
두 가지는 서로 다른 것이며 결과도 따로 저장된다
(`runs/<ID>/validation.json`).

---

## 9. 문서

| 파일 | 내용 |
|---|---|
| `README_KO.md` | (이 문서) 설치·실행·조회·재개 |
| `cortex_all_in_one.py` | 패키지 전체를 합친 **단일 파일 판** (설정 내장) |
| `tools/build_single_file.py` | 단일 파일을 다시 만드는 저작 도구 |
| `equations.md` | 실제 구현된 수식과 이산화, 자료형, 시간/단위 |
| `BIOLOGY_AND_ASSUMPTIONS.md` | 관찰 사실 / 계산 근사 / 미검증 가설 / 생략 범위 |
| `DATA_SCHEMA.md` | 3×3 인터페이스, 이벤트·상태·결과 스키마, 재현 범위 |
| `IMPLEMENTATION_STATUS.md` | 구현 여부와 검증 상태 (분리), 이번 실행은 전부 `not_run` |

---

## 10. 자주 나오는 문제

| 증상 | 원인과 해결 |
|---|---|
| `알 수 없는 설정 키: 'engine.dt_mss'` | 설정 키 오타다. 오타가 조용히 무시되지 않도록 일부러 오류를 낸다. `cortex/config.py` 의 `DEFAULTS` 에서 올바른 키를 확인하라 |
| `h5py 가 설치되어 있지 않다` | `python -m pip install -r requirements.txt` 또는 `recording.backend = "npz"` |
| `이미 완료된 실행 디렉터리다` | 결과를 덮어쓰지 않는다. 다른 `--run-dir` 를 주거나 기존 폴더를 옮겨라 |
| `설정 파일을 찾을 수 없다` | `configs/` 안의 파일 이름을 쓰거나 전체 경로를 입력하라 |
| `배선 규칙 ... 에서 시냅스 한도를 넘었다` | `wiring.max_total_synapses` 를 늘리거나 규칙의 `k` / `probability` 를 줄여라 |
| `이벤트 한도 초과` | `recording.max_events` 를 늘리거나 `recording.mode` 를 `selected`/`summary` 로 바꾸고 `selection_criterion` 을 적어라 |
| 그림의 한글이 네모로 나온다 | 한국어 글꼴(NanumGothic, Malgun Gothic 등)이 없는 환경이다. 축 라벨은 ASCII 이므로 정보는 읽힌다 |
| 하위 영역이 무반응처럼 보인다 | `engine.duration_ms` 가 신호가 도달하기에 너무 짧을 수 있다. `figures` 의 "영역별 신호 도달 시점" 그림을 먼저 확인하라 |

---

## 11. 이 구현이 **주장하지 않는** 것

* 인간 뇌의 완전하고 정확한 복제가 아니다.
* 측정하지 않은 속도를 "빠르다" 거나 "최적" 이라고 말하지 않는다.
* 고정 Gabor 대조 경로의 방향 선택성은 **학습된 것이 아니다**.
* 단안 영상을 좌/우에 복제한 입력은 **양안 시차 실험이 아니다**.
* 정지영상만 사용한 실행의 **운동 지표는 `not_applicable`** 이다.
* 조명 변화 조건의 검증 없이 **색채 항상성이 구현되었다고 말하지 않는다**.
* Rao 참조 모델은 전도도 LIF 회로와 **다른 엔진**이다. 동일 아키텍처 대조군인
  것처럼 섞어 순위를 매기지 않는다.
* Rao 모델은 `U` 의 전치를 쓰는 **기울기하강 모델**이다. "미분도 대칭 가중치도
  없는 학습" 이 아니다.
* 100만 화소 입력 처리는 입력 전처리의 규모이지 **100만 뉴런 학습이 아니다**.
