# singlefile_parts

`tools/build_single_file.py` 가 읽는 **조각 파일들**이다. 단독으로 import 되는
모듈이 아니라 `cortex_all_in_one.py` 안으로 그대로 들어가는 소스 조각이다.

| 파일 | 내용 |
|---|---|
| `builtin_configs.py` | 내장 설정용 공통 헬퍼(`_rule`, `_cortical_area`, `_local_microcircuit`, `_feedforward`, `_feedback`)와 세포 유형 표 |
| `builtin_minimal.py` | `configs/minimal.json` 을 그대로 옮긴 파이썬 리터럴 |
| `builtin_rest.py` | v1_small / hierarchy_small / megapixel_input 구성 함수와 `BUILTIN_CONFIGS` |
| `menu_and_entry.py` | 단일 파일용 한국어 메뉴, `selftest`, 진입점 |
| `patches/<태그>_old.txt` / `_new.txt` | 병합 후 치환할 원본/대체 텍스트 쌍 |

패치 태그:

| 태그 | 바꾸는 것 |
|---|---|
| a | matplotlib 모듈 수준 초기화 → `_plt()` 지연 로드 |
| b | `code_hash` 가 디렉터리 또는 단일 파일을 모두 처리 |
| c | 검증 14의 import 부작용 검사를 단일 파일 경로 import 로 |
| d | CLI 경로 상수(`PROJECT_ROOT`, `PACKAGE_ROOT`) |
| e, f | `--config` 도움말에 내장 설정 이름 안내 |
| g | `_load_cfg` 가 내장 설정 이름도 받도록 |
| h | argparse `prog` 이름 |
| i | 쓰이지 않는 `TYPE_CHECKING` import 제거 |

병합 결과가 원본 설정과 같은지는 `python cortex_all_in_one.py selftest` 로 확인한다.
