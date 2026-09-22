#!/usr/bin/env python3
"""make_manifest.py -- 모든 산출물의 해시 목록과 (선택) 압축 파일을 마지막에 생성한다.

명세 [16]-7, [16]-8.

사용법::

    python make_manifest.py
    python make_manifest.py --zip

``--zip`` 을 주면 ``results/neuron_matrix_v1_artifacts.zip`` 을 만들고
**압축 안의 실제 파일 목록**을 manifest 에 적는다. ZIP 자신의 해시는 ZIP 밖
(``results/HASHES.md`` 와 ``results/manifest.json``) 에 기록한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.persistence import file_sha256, hash_manifest

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}


def collect(root: str) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith((".pyc", ".zip")):
                continue
            out.append(os.path.join(dirpath, fn))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true", help="results/ 아래에 압축 파일도 만든다")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "manifest.json"))
    args = ap.parse_args()

    files = collect(HERE)
    code = [f for f in files if not f.startswith(os.path.join(HERE, "results"))]
    results = [f for f in files if f.startswith(os.path.join(HERE, "results"))]
    manifest = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": os.path.basename(HERE),
        "code_and_config": hash_manifest(code, HERE),
        "results": hash_manifest(results, HERE),
    }

    if args.zip:
        zpath = os.path.join(HERE, "results", "neuron_matrix_v1_artifacts.zip")
        os.makedirs(os.path.dirname(zpath), exist_ok=True)
        if os.path.exists(zpath):
            os.remove(zpath)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, os.path.relpath(f, HERE))
        with zipfile.ZipFile(zpath) as z:
            inner = [{"name": i.filename, "bytes": i.file_size, "crc32": f"{i.CRC:08x}"}
                     for i in z.infolist()]
        manifest["zip"] = {
            "path": os.path.relpath(zpath, HERE),
            "bytes": os.path.getsize(zpath),
            "sha256_recorded_outside_the_zip": file_sha256(zpath),
            "n_entries": len(inner),
            "entries": inner,
            "note_ko": "ZIP 안의 실제 파일 목록이다. ZIP 자신의 sha256 은 ZIP 밖인 이 파일과 HASHES.md 에 적는다.",
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    md = [f"# 산출물 해시 목록 (생성 {manifest['generated_at_utc']})", "",
          "sha256 는 전체 값이다. 경로는 프로젝트 루트 기준이다.", ""]
    for title, key in (("코드·설정", "code_and_config"), ("결과 파일", "results")):
        md += [f"## {title} ({len(manifest[key])}개)", "",
               "| 파일 | 바이트 | sha256 |", "|---|---|---|"]
        md += [f"| `{h['path']}` | {h['bytes']} | `{h['sha256']}` |" for h in manifest[key]]
        md += [""]
    if "zip" in manifest:
        z = manifest["zip"]
        md += ["## 압축 파일", "",
               f"- 경로: `{z['path']}` ({z['bytes']} 바이트, 항목 {z['n_entries']}개)",
               f"- **ZIP 자신의 sha256 (ZIP 밖에 기록): `{z['sha256_recorded_outside_the_zip']}`**",
               "", "압축 안의 실제 파일 목록은 `results/manifest.json` 의 `zip.entries` 에 있다.", ""]
    mdp = os.path.join(os.path.dirname(os.path.abspath(args.out)), "HASHES.md")
    with open(mdp, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"manifest: {args.out}")
    print(f"HASHES.md: {mdp}")
    print(f"  코드·설정 {len(manifest['code_and_config'])}개 / 결과 {len(manifest['results'])}개")
    if "zip" in manifest:
        print(f"  zip: {manifest['zip']['path']} ({manifest['zip']['n_entries']} entries)")
        print(f"  zip sha256 (밖에 기록): {manifest['zip']['sha256_recorded_outside_the_zip']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
