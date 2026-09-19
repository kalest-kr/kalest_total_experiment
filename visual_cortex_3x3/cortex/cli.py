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
    python -m cortex.cli run-all        --config v1_small --out D:/결과폴더

``run-all`` 은 **지정한 폴더에 전체 과정을 자동으로 실행**한다. 사용자가 출력
폴더를 명시해 직접 부르는 명령이므로 이 하나만 기본으로 실행되고, 계획만 보려면
``--dry-run`` 을 준다. 나머지 수치 실험 명령은 여전히 ``--execute`` 가 있어야
실행된다.
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


def _config_loader(spec: str) -> dict[str, Any]:
    """설정 이름/경로 -> 해석된 설정.

    단일 파일 판에서는 내장 설정 이름도 받는다 (그때는 전역에
    ``load_config_by_name_or_path`` 가 있다). 패키지 판에서는 JSON 경로다.
    """
    fn = globals().get("load_config_by_name_or_path") or globals().get("load_config")
    if fn is not None:
        return fn(spec)
    from .config import load
    return load(spec)


#: "전부" 를 뜻하는 표기들. ``*`` 는 셸이 펼치지 않고 그대로 들어오는 경우가
#: 많고(특히 Windows), 폴더 이름으로 쓸 수 없는 문자라 여기서 처리해야 한다.
_ALL_ALIASES = {"all", "*", "전부", "모두"}


def _expand_config_list(spec: str) -> list[str]:
    """쉼표로 구분한 설정 목록을 펼친다. ``all`` / ``*`` 는 쓸 수 있는 설정 전부."""
    out: list[str] = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() not in _ALL_ALIASES:
            out.append(item)
            continue
        builtin = globals().get("BUILTIN_CONFIGS")
        if builtin:
            out += sorted(builtin)
        else:
            cfg_dir = PROJECT_ROOT / "configs"
            out += [str(q) for q in sorted(cfg_dir.glob("*.json"))
                    if not q.name.startswith("_")]
    return out


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


def cmd_run_all(args: argparse.Namespace) -> int:
    """전체 과정을 자동 실행하고 지정한 폴더에 결과를 쓴다."""
    from .autorun import ALL_STAGES, run_everything

    configs = _expand_config_list(args.config)
    if not configs:
        print("[오류] --config 에 설정 이름이나 경로를 하나 이상 주어야 한다.",
              file=sys.stderr)
        return 2
    stages = [x.strip() for x in args.stages.split(",") if x.strip()] \
        if args.stages else list(ALL_STAGES)
    out = Path(args.out).expanduser()
    print("=" * 70)
    print(f"전체 자동 실행{' (dry-run)' if args.dry_run else ''}")
    print(f"  출력 폴더 : {out.resolve()}")
    print(f"  설정      : {configs}")
    print(f"  단계      : {stages}")
    print("  중단하려면 Ctrl+C. 중단해도 그때까지의 기록은 남는다.")
    print("=" * 70)
    try:
        summary = run_everything(
            out, configs, package_root=PACKAGE_ROOT, command=" ".join(sys.argv),
            backend=args.backend, stages=stages, limit_stimuli=args.limit_stimuli,
            duration_ms=args.duration_ms, seed=args.seed, dry_run=args.dry_run,
            overwrite=args.overwrite, stop_on_fail=args.stop_on_fail,
            config_loader=_config_loader, progress=None)
    except FileExistsError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    print(f"\n전체 상태: {summary['experiment_status']}")
    print(f"요약: {Path(summary['output_dir']) / 'summary.json'}")
    print(f"한국어 요약: {Path(summary['output_dir']) / 'SUMMARY_ko.md'}")
    return 0 if summary["all_stages_ok"] else 1


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

    sp = sub.add_parser(
        "run-all",
        help="전체 과정을 자동 실행하고 지정한 폴더에 결과를 쓴다 (기본으로 실행됨)")
    sp.add_argument("--out", required=True,
                    help="결과를 쓸 폴더 (없으면 만든다). 공백/한글 경로 가능")
    sp.add_argument("--config", default="minimal",
                    help="설정 이름/경로. 쉼표로 여러 개, 'all' 또는 '*' 이면 전부 "
                         "(기본: minimal)")
    sp.add_argument("--stages", default="",
                    help="실행할 단계 (쉼표 구분). 기본은 전부: "
                         "config_check,validate,simulate,experiment,reference,"
                         "report,figures")
    sp.add_argument("--backend", default="auto", choices=["auto", "hdf5", "npz"],
                    help="기록 백엔드. auto 는 h5py 가 없으면 npz 로 바꾸고 알린다")
    sp.add_argument("--limit-stimuli", type=int, default=0,
                    help="자극 수를 앞에서부터 N개로 제한 (0 이면 제한 없음)")
    sp.add_argument("--duration-ms", type=float, default=None,
                    help="engine.duration_ms 덮어쓰기")
    sp.add_argument("--seed", type=int, default=None, help="seeds.master 덮어쓰기")
    sp.add_argument("--overwrite", action="store_true",
                    help="이미 완료된 결과 폴더를 옆으로 옮기고 새로 쓴다")
    sp.add_argument("--stop-on-fail", action="store_true",
                    help="한 단계라도 실패하면 즉시 중단")
    sp.add_argument("--dry-run", action="store_true",
                    help="계획과 규모만 계산하고 실행하지 않는다")
    sp.set_defaults(func=cmd_run_all)

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
