"""cli.py -- 반복 작업용 명령행 인터페이스.

명세 14절. **수치 실험 명령은 기본이 dry-run** 이고 사용자가 ``--execute`` 를
명시할 때만 실행한다. 조회 명령(``report``, ``explain-neuron``, ``figures``)은
기존 기록만 읽으며 새 실험을 자동 실행하지 않는다.

메뉴(``run.py``)와 이 CLI 는 **같은 Runner/Recorder** 를 호출한다. 시뮬레이션
구현을 두 벌 만들지 않는다.

    python -m cortex.cli inspect-config --config configs/v1_small.json
    python -m cortex.cli validate       --config configs/minimal.json --execute
    python -m cortex.cli simulate       --config configs/v1_small.json --execute
    python -m cortex.cli experiment     --config configs/hierarchy_small.json --execute
    python -m cortex.cli resume         --run-dir runs/RUN_ID --execute
    python -m cortex.cli report         --run-dir runs/RUN_ID
    python -m cortex.cli explain-neuron --run-dir runs/RUN_ID --neuron-id 12 \
                                        --from-ms 0 --to-ms 100
    python -m cortex.cli figures        --run-dir runs/RUN_ID
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = PROJECT_ROOT / "runs"


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def _load_cfg(path: str) -> dict[str, Any]:
    from .config import ConfigError, load

    try:
        return load(path)
    except ConfigError as exc:
        print(f"[설정 오류] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _progress(msg: str) -> None:
    print(f"  … {msg}", flush=True)


def _runner(cfg: dict[str, Any], run_dir: Path | None, command: str,
            execute: bool) -> Any:
    from .runner import ExperimentRunner

    return ExperimentRunner(cfg, run_dir, command, PACKAGE_ROOT, execute=execute,
                            progress=_progress)


def _resolve_run_dir(args: argparse.Namespace, tag: str) -> Path:
    from .runner import new_run_dir

    if getattr(args, "run_dir", None):
        return Path(args.run_dir)
    root = Path(args.runs_root) if getattr(args, "runs_root", None) else DEFAULT_RUNS
    return new_run_dir(root, tag)


# ----------------------------------------------------------------------
def cmd_inspect_config(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    info = _runner(cfg, None, " ".join(sys.argv), False).inspect()
    _print_json(info)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    run_dir = _resolve_run_dir(args, f"validate_{cfg['meta']['name']}")
    r = _runner(cfg, run_dir, " ".join(sys.argv), args.execute)
    if not args.execute:
        print("[dry-run] 실행하지 않았다. 실제 검사를 돌리려면 --execute 를 붙여라.")
        _print_json(r.validate())
        return 0
    print(f"검사 실행 중... 결과는 {run_dir} 에 저장된다. 중단: Ctrl+C")
    result = r.validate()
    print(f"\n검사 {result['n_checks']}건: 통과 {result['n_passed']} / "
          f"실패 {result['n_failed']} / 건너뜀 {result['n_skipped']}")
    for c in result["checks"]:
        print(f"  [{c['status'].upper():7s}] {c['id']:02d} {c['name']}")
        if c["status"] != "passed":
            for a in c["assertions"]:
                if not a["passed"]:
                    print(f"      - 실패: {a['assertion']}")
            if c["reason"]:
                print(f"      - 사유: {c['reason']}")
    print(f"\n결과 파일: {run_dir / 'validation.json'}")
    return 0 if result["n_failed"] == 0 else 1


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    run_dir = _resolve_run_dir(args, f"simulate_{cfg['meta']['name']}")
    r = _runner(cfg, run_dir, " ".join(sys.argv), args.execute)
    if not args.execute:
        print("[dry-run] 규모만 계산했다. 실행하려면 --execute 를 붙여라.")
        _print_json(r.simulate())
        return 0
    print(f"시뮬레이션 실행 중... 결과: {run_dir}  (중단: Ctrl+C)")
    _print_json(r.simulate())
    print(f"완료. 기록 위치: {run_dir}")
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    run_dir = _resolve_run_dir(args, f"experiment_{cfg['meta']['name']}")
    r = _runner(cfg, run_dir, " ".join(sys.argv), args.execute)
    if not args.execute:
        print("[dry-run] 규모만 계산했다. 실행하려면 --execute 를 붙여라.")
        _print_json(r.experiment())
        return 0
    print(f"실험 실행 중... 결과: {run_dir}  (중단: Ctrl+C)")
    _print_json(r.experiment())
    print(f"완료. 기록 위치: {run_dir}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    from .recording import RecordingError, load_manifest

    run_dir = Path(args.run_dir)
    try:
        man = load_manifest(run_dir)
    except RecordingError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    cfg = man["config"]
    r = _runner(cfg, run_dir, " ".join(sys.argv), args.execute)
    try:
        out = r.resume(Path(args.checkpoint) if args.checkpoint else None,
                       allow_mismatch=args.allow_mismatch)
    except RecordingError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    _print_json(out)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .analysis import make_report, verify_report_matches_records
    from .recording import RecordingError

    try:
        path = make_report(Path(args.run_dir))
    except RecordingError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    print(f"보고서 생성: {path}")
    check = verify_report_matches_records(Path(args.run_dir))
    print(f"보고서-기록 일치 검사: {check}")
    return 0


def cmd_explain_neuron(args: argparse.Namespace) -> int:
    from .analysis import explain_neuron

    out = explain_neuron(Path(args.run_dir), args.neuron_id, args.from_ms, args.to_ms,
                         top_k=args.top_k)
    _print_json(out)
    return 0 if out.get("found") else 1


def cmd_figures(args: argparse.Namespace) -> int:
    from .recording import RecordingError, load_manifest
    from .visualization import make_all

    run_dir = Path(args.run_dir)
    try:
        load_manifest(run_dir)
    except RecordingError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    model = None
    if args.rebuild_model:
        from . import rng as rng_mod
        from .runner import build_model
        man = load_manifest(run_dir)
        model = build_model(man["config"], rng_mod.from_config(man["config"]))
    nids = [int(x) for x in args.neuron_ids.split(",")] if args.neuron_ids else []
    made = make_all(run_dir, model, nids)
    print(f"그림 {len(made)}개 생성:")
    for p in made:
        print(f"  {p}")
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    root = Path(args.runs_root) if args.runs_root else DEFAULT_RUNS
    if not root.is_dir():
        print(f"실행 기록 폴더가 없다: {root}")
        return 0
    rows = []
    for d in sorted(root.iterdir()):
        man = d / "manifest.json"
        if man.is_file():
            m = json.loads(man.read_text(encoding="utf-8"))
            rows.append((d.name, m.get("status"), m.get("started_utc"),
                         m.get("config", {}).get("meta", {}).get("name")))
    if not rows:
        print(f"{root} 안에 실행 기록이 없다.")
        return 0
    print(f"{'run_id':45s} {'status':12s} {'started(UTC)':22s} config")
    for r in rows:
        print(f"{r[0]:45s} {str(r[1]):12s} {str(r[2]):22s} {r[3]}")
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cortex.cli",
        description="3x3 뉴런 기록 구조 시각피질 시뮬레이터 CLI "
                    "(수치 실험은 --execute 를 붙여야 실행된다)")
    sub = p.add_subparsers(dest="command", required=True)

    def add_cfg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", required=True, help="설정 JSON 경로")
        sp.add_argument("--run-dir", default=None, help="출력 폴더 (기본: runs/<자동>)")
        sp.add_argument("--runs-root", default=None, help="runs 루트 폴더")
        sp.add_argument("--execute", action="store_true",
                        help="실제로 실행한다 (없으면 규모만 계산하는 dry-run)")

    sp = sub.add_parser("inspect-config", help="설정과 규모만 확인 (실행하지 않음)")
    sp.add_argument("--config", required=True)
    sp.set_defaults(func=cmd_inspect_config)

    sp = sub.add_parser("validate", help="필수 검증 1~14 실행")
    add_cfg(sp)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("simulate", help="자극 제시 시뮬레이션")
    add_cfg(sp)
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("experiment", help="train/dev/test 분할 실험")
    add_cfg(sp)
    sp.set_defaults(func=cmd_experiment)

    sp = sub.add_parser("resume", help="중단된 실행 재개")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--allow-mismatch", action="store_true",
                    help="코드/설정 해시가 달라도 재개 (차이를 인지한 경우에만)")
    sp.add_argument("--execute", action="store_true")
    sp.set_defaults(func=cmd_resume)

    sp = sub.add_parser("report", help="기존 기록에서 한국어 보고서 생성")
    sp.add_argument("--run-dir", required=True)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("explain-neuron", help="특정 뉴런의 입력·발화·출력 조회")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--neuron-id", type=int, required=True)
    sp.add_argument("--from-ms", type=float, default=0.0)
    sp.add_argument("--to-ms", type=float, default=float("inf"))
    sp.add_argument("--top-k", type=int, default=20)
    sp.set_defaults(func=cmd_explain_neuron)

    sp = sub.add_parser("figures", help="기존 기록에서 그림 생성")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--neuron-ids", default="", help="쉼표 구분 뉴런 ID")
    sp.add_argument("--rebuild-model", action="store_true",
                    help="manifest 의 설정으로 모델을 다시 조립해 배치/지도 그림도 만든다")
    sp.set_defaults(func=cmd_figures)

    sp = sub.add_parser("list-runs", help="실행 기록 목록")
    sp.add_argument("--runs-root", default=None)
    sp.set_defaults(func=cmd_list_runs)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 Ctrl+C 로 중단했다. "
              "기록은 runs/<run_id>/ 에 저장되어 있고 resume 으로 재개할 수 있다.")
        return 130
    except FileNotFoundError as exc:
        print(f"[오류] 파일을 찾을 수 없다: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
