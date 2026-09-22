"""build_single_file.py -- cortex 패키지를 단일 파일로 병합하는 **저작 도구**.

사용법::

    python tools/build_single_file.py

`cortex/` 22개 모듈을 의존성 순서로 이어 붙이고, 상대 import 제거·이름 충돌
해소·matplotlib 지연 로드·내장 설정 삽입을 적용해 `cortex_all_in_one.py` 를
만든다. 실제로 바꾼 내용은 `singlefile_parts/patches/` 의 old/new 조각에 있다.

**이 스크립트는 시뮬레이션이나 학습을 실행하지 않는다.** 결과 파일이 원본과
같은지는 `python cortex_all_in_one.py selftest` 로 확인한다.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
OUT = SRC / "cortex_all_in_one.py"

ORDER = [
    ("units", "단위 규약과 물리 상수"),
    ("ids", "안정적인 정수 ID 와 이름 레지스트리"),
    ("config", "설정 스키마, 기본값, 병합과 검증"),
    ("rng", "이름이 고정된 독립 난수 스트림"),
    ("records", "3x3 뉴런 기록 인터페이스와 타입 배열"),
    ("events", "입력 이벤트 로그와 지연 사건 큐"),
    ("synapses", "희소 edge table 과 인덱스"),
    ("retina", "영상 입력, 색 변환, DoG, ON/OFF, 입력 구동"),
    ("retinotopy", "불균일(로그-극좌표) 샘플링과 좌표 변환"),
    ("anatomy", "3D 배치, 층, 세포 유형, 방향/안구우세 지도"),
    ("areas", "영역 간/영역 내 배선 생성"),
    ("dynamics", "공통 시간 루프와 두 동작 모드"),
    ("plasticity", "국소 가소성(STDP)과 항상성 임계 적응"),
    ("predictive_coding", "Rao 계열 연속값 예측 부호화 참조 모델"),
    ("stimuli", "자극 생성기"),
    ("v1_reference", "고정 Gabor 대조 경로와 위상 불변 에너지"),
    ("recording", "실행 기록기, 저장소 백엔드, 체크포인트"),
    ("runner", "모델 조립과 실험 실행기"),
    ("analysis", "저장된 기록에서만 읽는 분석·조회·보고서"),
    ("visualization", "저장된 기록을 읽어 그림 생성"),
    ("validation", "필수 검증 1~14"),
    ("autorun", "전체 자동 실행 오케스트레이션"),
    ("cli", "명령행 인터페이스"),
]

# (모듈, 원래 이름, 새 이름)
RENAMES = [
    ("anatomy", "build", "build_anatomy"),
    ("areas", "build", "build_wiring"),
    ("plasticity", "make", "make_plasticity"),
    ("predictive_coding", "make", "make_rao_model"),
    ("rng", "from_config", "rng_from_config"),
    ("stimuli", "generate", "generate_stimuli"),
    ("config", "load", "load_config"),
    ("config", "resolve", "resolve_config"),
    ("config", "validate", "validate_config"),
    ("cli", "main", "cli_main"),
]

# 모듈 별칭 호출 -> 새 이름
ALIAS_CALLS = {
    "anatomy_mod.build(": "build_anatomy(",
    "areas_mod.build(": "build_wiring(",
    "plasticity_mod.make(": "make_plasticity(",
    "rng_mod.from_config(": "rng_from_config(",
    "stimuli_mod.generate(": "generate_stimuli(",
    # config 모듈은 이름이 바뀌므로 괄호 없이도 바꾼다 (함수 객체로 넘기는 곳이 있다).
    "config_mod.load": "load_config",
    "config_mod.resolve": "resolve_config",
    "config_mod.validate": "validate_config",
}
ALIAS_PREFIX = re.compile(
    r"\b(anatomy_mod|areas_mod|config_mod|plasticity_mod|rng_mod|stimuli_mod"
    r"|validation_mod)\.")

REL_IMPORT_LINE = re.compile(r"^\s*from \.[A-Za-z_]* import .*$|^\s*from \. import .*$")


def _is_main_guard(node: ast.If) -> bool:
    t = node.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
            and t.left.id == "__name__"
            and len(t.comparators) == 1
            and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value == "__main__")


def extract(module: str) -> tuple[str, str, list[str]]:
    """(docstring, 본문 코드, 절대 import 목록)."""
    path = SRC / "cortex" / f"{module}.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(lines))
    drop: list[tuple[int, int]] = []      # 1-based inclusive
    imports: list[str] = []
    doc = ""

    for i, node in enumerate(tree.body):
        lo, hi = node.lineno, (node.end_lineno or node.lineno)
        if i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            doc = node.value.value
            drop.append((lo, hi))
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module == "__future__":
                drop.append((lo, hi))
            else:
                names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "")
                                  for a in node.names)
                imports.append(f"from {node.module} import {names}")
                drop.append((lo, hi))
            continue
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.append(f"import {a.name}" + (f" as {a.asname}" if a.asname else ""))
            drop.append((lo, hi))
            continue
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            drop.append((lo, hi))
            continue
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) \
                and node.test.id == "TYPE_CHECKING":
            drop.append((lo, hi))
            continue
        if isinstance(node, ast.If) and _is_main_guard(node):
            # 각 모듈의 `if __name__ == "__main__":` 은 단일 파일에서 하나만 둔다
            drop.append((lo, hi))
            continue
        if isinstance(node, ast.Try):
            # scipy / h5py 가드 import 블록은 본문에 남기되 절대 import 는 유지한다
            continue

    dropped = set()
    for lo, hi in drop:
        dropped.update(range(lo, hi + 1))
    body = [ln for n, ln in enumerate(lines, start=1) if n not in dropped]
    # 함수 안의 상대 import 제거
    body = [ln for ln in body if not REL_IMPORT_LINE.match(ln)]
    text = "\n".join(body).strip("\n")
    return doc, text, imports


def apply_renames(module: str, text: str) -> str:
    for mod, old, new in RENAMES:
        if mod != module:
            continue
        text = re.sub(rf"^def {old}\(", f"def {new}(", text, flags=re.M)
    return text


def fix_references(text: str) -> str:
    for old, new in ALIAS_CALLS.items():
        text = text.replace(old, new)
    text = ALIAS_PREFIX.sub("", text)
    # config 내부 호출부
    text = text.replace("    cfg = resolve(raw)", "    cfg = resolve_config(raw)")
    text = text.replace("    validate(cfg)\n    return cfg", "    validate_config(cfg)\n    return cfg")
    text = text.replace("        return load(path)", "        return load_config(path)")
    return text


SCRATCH = Path(__file__).resolve().parent / "singlefile_parts"


def read_scratch(name: str) -> str:
    return (SCRATCH / name).read_text(encoding="utf-8")


def _pair(tag: str) -> tuple[str, str]:
    d = SCRATCH / "patches"
    return ((d / f"{tag}_old.txt").read_text(encoding="utf-8"),
            (d / f"{tag}_new.txt").read_text(encoding="utf-8"))


CONFIG_HELP_DECL = (
    'CONFIG_HELP = ("\u002d\u002dconfig: \uc124\uc815 \uc774\ub984 \ub610\ub294 '
    '\uacbd\ub85c")\n')


def single_file_patches(out: str) -> str:
    """단일 파일용 조정 (원본과의 차이는 파일 머리말에 적혀 있다)."""
    for tag in ("a", "b", "c", "d", "e", "f", "g", "h", "i"):
        old, new = _pair(tag)
        old = old.rstrip("\n")
        if tag != "a":
            new = new.rstrip("\n")
        if old not in out:
            raise SystemExit(f"패치 {tag}: 대상 텍스트를 찾지 못했다")
        out = out.replace(old, new, 1)

    # visualization 섹션 안의 plt. -> _plt().
    start = out.index("# 섹션: visualization")
    end = out.index("# 섹션: validation")
    vis = out[start:end]
    vis = re.sub(r"(?<![\w.])plt\.", "_plt().", vis)
    out = out[:start] + vis + out[end:]

    # CLI 도움말 상수 삽입
    anchor = 'DEFAULT_RUNS = PROJECT_ROOT / "runs"'
    help_const = (anchor + "\n\n"
                  "#: --config 인자 도움말 (내장 설정 이름 또는 JSON 경로)\n"
                  'CONFIG_HELP = ("내장 설정 이름(minimal, v1_small, hierarchy_small, "\n'
                  '               "megapixel_input) 또는 설정 JSON 파일 경로")')
    out = out.replace(anchor, help_const, 1)
    return out


def main() -> None:
    sections: list[tuple[str, str, str]] = []
    all_imports: list[str] = []
    for module, title in ORDER:
        doc, text, imports = extract(module)
        text = apply_renames(module, text)
        sections.append((module, title, (doc, text)))
        all_imports.extend(imports)


    # matplotlib 은 지연 로드로 바꾼다 (import 부작용 금지)
    all_imports = [i for i in all_imports if "matplotlib" not in i]
    plain = sorted({i for i in all_imports if i.startswith("import ")})
    by_mod: dict[str, set[str]] = {}
    for i in all_imports:
        if not i.startswith("from ") or i.startswith("from __future__"):
            continue
        head, _, names = i.partition(" import ")
        mod = head[len("from "):]
        by_mod.setdefault(mod, set()).update(n.strip() for n in names.split(","))

    THIRD = {"numpy", "scipy", "PIL", "matplotlib", "h5py"}

    def is_third(mod: str) -> bool:
        return mod.split(".")[0] in THIRD

    std_plain = [i for i in plain if not is_third(i.split()[1])]
    third_plain = [i for i in plain if is_third(i.split()[1])]
    std_from = [f"from {m} import " + ", ".join(sorted(by_mod[m]))
                for m in sorted(by_mod) if not is_third(m)]
    third_from = [f"from {m} import " + ", ".join(sorted(by_mod[m]))
                  for m in sorted(by_mod) if is_third(m)]

    parts: list[str] = [HEADER]
    parts.append("from __future__ import annotations\n")
    parts.append("# --- 표준 라이브러리 ---")
    parts.append("\n".join(std_plain + std_from))
    parts.append("")
    parts.append("# --- 서드파티 (matplotlib 은 _plt() 에서 지연 로드) ---")
    parts.append("\n".join(third_plain + third_from))
    parts.append("\n" + TOC)

    for module, title, (doc, text) in sections:
        banner = ("\n" + "# " + "=" * 76 + "\n"
                  f"# 섹션: {module}  —  {title}\n"
                  f"#   (원래 파일: cortex/{module}.py)\n"
                  + "# " + "=" * 76 + "\n")
        parts.append(banner)
        if doc:
            parts.append('"""' + doc.rstrip() + '\n"""\n')
        parts.append(text + "\n")

    parts.append(read_scratch("builtin_configs.py"))
    parts.append(read_scratch("builtin_minimal.py"))
    parts.append(read_scratch("builtin_rest.py"))
    parts.append("\n")
    parts.append(read_scratch("menu_and_entry.py"))

    out = "\n".join(parts)
    out = fix_references(out)
    out = single_file_patches(out)
    OUT.write_text(out, encoding="utf-8")
    print(f"wrote {OUT} ({len(out.splitlines())} lines, {len(out)} bytes)")




HEADER = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cortex_all_in_one.py — 3x3 뉴런 기록 구조 시각피질 시뮬레이터 (단일 파일 판)

이 파일 **하나만** 있으면 동작한다. 원래 패키지(`cortex/` 22개 모듈 + `run.py`)와
같은 코드를 순서대로 이어 붙이고, 설정 4종을 코드로 내장했다.

    python cortex_all_in_one.py                     # 한국어 메뉴
    python cortex_all_in_one.py inspect-config --config minimal
    python cortex_all_in_one.py validate --config minimal --execute
    python cortex_all_in_one.py simulate --config v1_small --execute
    python cortex_all_in_one.py report --run-dir runs/RUN_ID

`--config` 에는 내장 설정 이름(minimal, v1_small, hierarchy_small,
megapixel_input) 또는 JSON 파일 경로를 줄 수 있다.

=== 이번 납품의 수치 실험 상태는 전부 `not_run` 이다 ===
이 파일을 만드는 동안 학습·시뮬레이션·데이터 생성·수치 검증·그래프 생성을 한 번도
실행하지 않았다. 실행은 사용자가 명령을 내릴 때만 일어난다. 자세한 내용은
같은 폴더의 IMPLEMENTATION_STATUS.md 참조.

--- 병합하면서 바꾼 것 (원본과의 차이) ---
1. 상대 import(`from .x import y`)를 모두 제거했다. 모든 이름이 한 모듈 안에 있다.
2. 이름이 겹치던 최상위 함수 2쌍을 구분되게 바꿨다:
     anatomy.build          -> build_anatomy
     areas.build            -> build_wiring
     plasticity.make        -> make_plasticity
     predictive_coding.make -> make_rao_model
   읽기 쉽도록 다음도 함께 바꿨다:
     rng.from_config -> rng_from_config      stimuli.generate -> generate_stimuli
     config.load     -> load_config          config.resolve   -> resolve_config
     config.validate -> validate_config
3. matplotlib 은 **지연 로드**(`_plt()`)로 바꿨다. 이 파일을 import 만 해도
   백엔드 초기화가 일어나지 않는다 (import 부작용 금지 규칙 유지).
4. 설정 4종을 `BUILTIN_CONFIGS` 로 내장했다. JSON 파일 없이도 실행된다.
5. 코드 해시(`code_hash`)는 디렉터리 대신 **이 파일 하나**를 해싱한다.
6. import 부작용 검사(검증 14)는 이 파일을 경로로 import 하는 방식으로 바꿨다.

그 외 계산 로직, 자료구조, 검증 기준, 단위 규약은 원본과 같다.
각 섹션의 제목은 원래 모듈 이름이며, 섹션 첫머리의 문자열이 그 모듈의 설명이다.
"""
'''

TOC = '''# ----------------------------------------------------------------------------
# 목차 (원래 모듈 순서 = 의존성 순서)
#   01 units              단위 규약           12 dynamics          시간 루프/두 모드
#   02 ids                ID 레지스트리       13 plasticity        STDP/항상성
#   03 config             설정 스키마         14 predictive_coding Rao 참조 모델
#   04 rng                난수 스트림         15 stimuli           자극 생성
#   05 records            3x3 기록 인터페이스 16 v1_reference      고정 Gabor 대조
#   06 events             이벤트/지연 큐      17 recording         기록기/체크포인트
#   07 synapses           희소 edge table     18 runner            실험 실행기
#   08 retina             색/DoG/ON-OFF       19 analysis          조회/보고서
#   09 retinotopy         로그-극좌표         20 visualization     그림
#   10 anatomy            3D 배치/지도        21 validation        필수 검증 1~14
#   11 areas              배선 생성           22 cli               명령행 인터페이스
#   23 builtin configs    내장 설정 4종       24 menu              한국어 터미널 메뉴
# ----------------------------------------------------------------------------
'''

BUILTIN_CONFIG_SECTION = "@@BUILTIN@@"
ENTRYPOINT = "@@ENTRY@@"

if __name__ == "__main__":
    main()
