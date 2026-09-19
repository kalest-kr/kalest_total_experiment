#!/usr/bin/env python3
"""build_zip.py -- 배포용 ZIP 과 체크섬 파일을 만든다.

**시뮬레이션을 실행하지 않는다.** 파일을 모아 압축하고 해시를 적을 뿐이다.

포함 대상은 git 이 추적하는 `visual_cortex_3x3/` 아래 파일이다. 실행 기록
(`runs/`), 캐시(`__pycache__`), 그림·데이터 산출물은 들어가지 않는다.

ZIP 자신의 sha256 은 **ZIP 밖의** `visual_cortex_3x3_code.CHECKSUMS.txt` 에 적는다
(안에 넣으면 자기 자신을 해싱하는 모순이 생긴다).

사용법::

    python tools/build_zip.py
"""

from __future__ import annotations

import hashlib
import subprocess
import time
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent      # visual_cortex_3x3/
REPO = PROJECT.parent
ZIP_PATH = REPO / "visual_cortex_3x3_code.zip"
SUM_PATH = REPO / "visual_cortex_3x3_code.CHECKSUMS.txt"

#: 재현 가능한 ZIP 을 위해 고정한 타임스탬프 (UTC). 파일 내용이 같으면
#: ZIP 도 바이트 단위로 같아진다.
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z", "--", str(PROJECT.name)],
                         cwd=REPO, capture_output=True, text=True, check=True)
    names = [n for n in out.stdout.split("\0") if n]
    paths = [REPO / n for n in names]
    keep = [p for p in paths
            if p.is_file() and "__pycache__" not in p.parts and "runs" not in p.parts]
    return sorted(keep, key=lambda p: str(p.relative_to(REPO)))


def main() -> int:
    files = tracked_files()
    if not files:
        raise SystemExit("git 이 추적하는 파일이 없다. 먼저 git add 를 하라.")

    rows: list[tuple[str, int, str, str]] = []
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            arc = str(f.relative_to(REPO)).replace("\\", "/")
            data = f.read_bytes()
            info = zipfile.ZipInfo(arc, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
            rows.append((arc, len(data),
                         f"{zipfile.crc32(data) & 0xFFFFFFFF:08x}",
                         hashlib.sha256(data).hexdigest()))

    zip_bytes = ZIP_PATH.read_bytes()
    lines = [
        "# visual_cortex_3x3 코드 ZIP 체크섬 (ZIP 밖에 기록)",
        "",
        f"- 생성(UTC): {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"- 파일: {ZIP_PATH.name} ({len(zip_bytes)} 바이트, 항목 {len(rows)}개)",
        "- **ZIP 자신의 sha256 (이 파일은 ZIP 안에 들어 있지 않다):**",
        f"  `{hashlib.sha256(zip_bytes).hexdigest()}`",
        "",
        "이 ZIP 에는 **실행 결과가 들어 있지 않다.** `runs/` 디렉터리는 없다.",
        "무엇을 실제로 실행했고 무엇을 실행하지 않았는지는",
        "`visual_cortex_3x3/IMPLEMENTATION_STATUS.md` 0절에 적혀 있다.",
        "",
        "단일 파일 판은 `visual_cortex_3x3/cortex_all_in_one.py` 이며, 패키지와 같은",
        "코드다 (차이는 README_KO.md 2-1절과 파일 머리말 참조).",
        "",
        "전체를 한 번에 실행하고 원하는 폴더에 결과를 모으려면::",
        "",
        "    python -m cortex.cli run-all --config minimal --out D:/결과폴더",
        "",
        "## ZIP 안의 실제 파일 목록",
        "",
        "| 경로 | 바이트 | crc32 | sha256 |",
        "|---|---|---|---|",
    ]
    lines += [f"| `{a}` | {n} | `{c}` | `{h}` |" for a, n, c, h in rows]
    lines.append("")
    SUM_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {ZIP_PATH} ({len(zip_bytes)} bytes, {len(rows)} entries)")
    print(f"wrote {SUM_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
