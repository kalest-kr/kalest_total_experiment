# ============================================================================
# 섹션: menu  —  한국어 터미널 메뉴 (원래 파일: run.py)
# ============================================================================
"""한국어 터미널 메뉴.

**사용자가 메뉴에서 고르기 전에는 시뮬레이션이나 학습을 시작하지 않는다.**
메뉴를 고르는 것은 명시적인 실행 요청이므로 그때 해당 작업을 수행하고
진행률·중단 방법·결과 위치를 보여준다. 종료할 때 다른 실험을 자동으로
이어서 돌리지 않는다.

기본 경로는 이 파일 위치를 기준으로 해석하므로 다른 작업 디렉터리에서
실행해도 동작한다. 경로에 공백이나 한글이 있어도 된다.
"""

MENU = """
==================================================================
  3x3 뉴런 기록 구조 시각피질 시뮬레이터 (단일 파일 판)
  (망막 - LGN - V1 - V2 - V3 - V4 - IT, 연구용)
==================================================================
  A) 전체 자동 실행 — 내가 지정한 폴더에 모든 결과를 쓴다
  1) 최소 모델 검증 실행           (내장 설정 minimal)
  2) V1 시뮬레이션 실행             (내장 설정 v1_small)
  3) 전체 시각 경로 작은 모델 실험  (내장 설정 hierarchy_small)
  4) 기존 실행 기록에서 보고서·그래프 생성
  5) 특정 뉴런의 입력·발화·출력 로그 조회
  6) 중단된 실험 재개
  7) 설정만 확인 (규모 추정, 실행하지 않음)
  8) 실행 기록 목록 보기
  9) 내장 설정 자체 점검 (configs/*.json 이 있으면 비교)
  0) 종료
------------------------------------------------------------------
  A) 를 고르면 설정 확인 -> 검증 -> 시뮬레이션 -> 실험 -> 참조 모델
  -> 보고서 -> 그림 까지 한 번에 돌리고 결과를 지정 폴더에 정리한다.
==================================================================
"""

PRESETS: dict[str, tuple[str, str, str]] = {
    "1": ("minimal", "validate", "최소 모델 검증"),
    "2": ("v1_small", "simulate", "V1 시뮬레이션"),
    "3": ("hierarchy_small", "experiment", "전체 시각 경로 작은 모델 실험"),
}


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return val or default


def _check_dependencies() -> bool:
    missing: list[str] = []
    for mod, pkg in (("numpy", "numpy"), ("scipy", "scipy"),
                     ("PIL", "Pillow"), ("matplotlib", "matplotlib")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n[필수 패키지가 없다]")
        print(f"  없는 패키지: {', '.join(missing)}")
        print("  아래 명령을 먼저 실행하라 (사용자 환경에서 직접 실행):")
        print("    python -m pip install numpy scipy Pillow matplotlib h5py")
        return False
    try:
        __import__("h5py")
    except ImportError:
        print("\n[알림] h5py 가 설치되어 있지 않다.")
        print("  대용량 기록(events.h5, states.h5)을 쓰려면 설치가 필요하다:")
        print("    python -m pip install h5py")
        print("  설치하지 않으려면 설정에서 recording.backend 를 \"npz\" 로 바꿔라.")
    return True


def _resolve_config_choice(name_or_path: str) -> str | None:
    """내장 설정 이름 또는 파일 경로를 확인해 그대로 돌려준다."""
    key = str(name_or_path).strip()
    if key in BUILTIN_CONFIGS or Path(key).stem in BUILTIN_CONFIGS:
        return key
    if Path(key).expanduser().is_file():
        return str(Path(key).expanduser())
    print(f"\n[설정을 찾을 수 없다] {name_or_path}")
    print(f"  내장 설정 이름: {', '.join(sorted(BUILTIN_CONFIGS))}")
    cfg_dir = PROJECT_ROOT / "configs"
    if cfg_dir.is_dir():
        print(f"  또는 {cfg_dir} 안의 JSON 파일:")
        for f in sorted(cfg_dir.glob("*.json")):
            print(f"    - {f.name}")
    print("  전체 경로를 직접 입력해도 된다 (공백/한글 경로 가능).")
    return None


def _run_preset(choice: str) -> None:
    name, action, title = PRESETS[choice]
    cfg = _resolve_config_choice(_ask("설정 (내장 이름 또는 파일 경로)", name))
    if cfg is None:
        return
    out = _ask("출력 폴더 (비우면 자동 생성)", "")
    print(f"\n[{title}] 을(를) 실행한다.")
    print("  중단하려면 Ctrl+C 를 누르면 된다. 중단해도 기록과 체크포인트는 남는다.")
    argv = [action, "--config", cfg, "--execute"]
    if out:
        argv += ["--run-dir", out]
    else:
        argv += ["--runs-root", str(DEFAULT_RUNS)]
    code = cli_main(argv)
    print(f"\n[완료] 종료 코드 {code}. 결과는 {out or DEFAULT_RUNS} 아래에 있다.")


def _menu_report() -> None:
    run_dir = _ask("실행 기록 폴더 (runs/<run_id>)", "")
    if not run_dir:
        print("  실행 기록 폴더가 필요하다. 메뉴 8) 로 목록을 먼저 확인하라.")
        return
    if not Path(run_dir).is_dir():
        print(f"\n[폴더가 없다] {run_dir}")
        print("  메뉴 8) 로 실행 기록 목록을 확인하고 정확한 경로를 입력하라.")
        return
    cli_main(["report", "--run-dir", run_dir])
    nid = _ask("그림에 포함할 뉴런 ID (쉼표 구분, 비우면 생략)", "")
    argv = ["figures", "--run-dir", run_dir]
    if nid:
        argv += ["--neuron-ids", nid]
    if _ask("배치/지도 그림도 만들까? 모델을 다시 조립한다 (y/N)", "N").lower() == "y":
        argv += ["--rebuild-model"]
    cli_main(argv)
    print(f"\n[완료] 보고서와 그림은 {run_dir} 아래에 있다.")


def _menu_explain_neuron() -> None:
    run_dir = _ask("실행 기록 폴더", "")
    if not run_dir or not Path(run_dir).is_dir():
        print("  올바른 실행 기록 폴더가 필요하다. 메뉴 8) 로 목록을 확인하라.")
        return
    nid = _ask("뉴런 ID", "0")
    t0 = _ask("시작 시각 [ms]", "0")
    t1 = _ask("종료 시각 [ms]", "100")
    try:
        cli_main(["explain-neuron", "--run-dir", run_dir,
                  "--neuron-id", str(int(nid)), "--from-ms", str(float(t0)),
                  "--to-ms", str(float(t1))])
    except ValueError:
        print("  숫자를 입력해야 한다 (뉴런 ID 는 정수, 시각은 실수).")


def _menu_resume() -> None:
    run_dir = _ask("재개할 실행 기록 폴더", "")
    if not run_dir or not Path(run_dir).is_dir():
        print("  올바른 실행 기록 폴더가 필요하다. 메뉴 8) 로 목록을 확인하라.")
        return
    print("  재개한다. 중단하려면 Ctrl+C.")
    cli_main(["resume", "--run-dir", run_dir, "--execute"])


def _menu_run_all() -> None:
    """전체 과정을 자동 실행하고 사용자가 지정한 폴더에 결과를 쓴다."""
    out = _ask("결과를 쓸 폴더 (예: D:/결과폴더, 공백/한글 가능)", "")
    if not out:
        print("  결과 폴더를 반드시 입력해야 한다. 아무 것도 실행하지 않았다.")
        return
    spec = _ask("설정 (내장 이름/경로, 쉼표로 여러 개, 'all' 이면 내장 전부)",
                "minimal")
    stages = _ask("실행할 단계 (쉼표 구분, 비우면 전부)", "")
    limit = _ask("자극 수 제한 (0 이면 제한 없음)", "0")
    dry = _ask("계획만 보고 실행은 하지 않을까? (y/N)", "N").lower() == "y"
    try:
        limit_n = int(limit)
    except ValueError:
        print("  자극 수 제한은 정수여야 한다.")
        return
    argv = ["run-all", "--out", out, "--config", spec,
            "--limit-stimuli", str(limit_n)]
    if stages:
        argv += ["--stages", stages]
    if dry:
        argv += ["--dry-run"]
    print("\n[전체 자동 실행] 을(를) 시작한다.")
    print("  중단하려면 Ctrl+C 를 누르면 된다. 중단해도 그때까지의 기록은 남는다.")
    code = cli_main(argv)
    print(f"\n[완료] 종료 코드 {code}. 결과는 {out} 아래에 있다.")
    print(f"  요약: {Path(out) / 'SUMMARY_ko.md'}")


def _menu_inspect() -> None:
    cfg = _resolve_config_choice(_ask("설정 (내장 이름 또는 파일 경로)", "v1_small"))
    if cfg is None:
        return
    cli_main(["inspect-config", "--config", cfg])
    print("\n[알림] 위 값은 실행 전 추정이다. 실제 런타임은 측정 전에 확정하지 않는다.")


def menu_main() -> int:
    print(MENU)
    if not _check_dependencies():
        return 2
    while True:
        try:
            choice = _ask("번호를 선택하라", "0")
        except KeyboardInterrupt:
            print("\n종료한다.")
            return 0
        try:
            if choice == "0":
                print("종료한다. (다른 실험을 자동으로 실행하지 않는다.)")
                return 0
            if choice in ("a", "A"):
                _menu_run_all()
            elif choice in PRESETS:
                _run_preset(choice)
            elif choice == "4":
                _menu_report()
            elif choice == "5":
                _menu_explain_neuron()
            elif choice == "6":
                _menu_resume()
            elif choice == "7":
                _menu_inspect()
            elif choice == "8":
                cli_main(["list-runs", "--runs-root", str(DEFAULT_RUNS)])
            elif choice == "9":
                selftest()
            else:
                print(f"  '{choice}' 은(는) 없는 번호다. "
                      f"A 또는 0~9 중에서 고르라.")
        except KeyboardInterrupt:
            print("\n[중단] 작업을 중단했다. 기록은 runs/ 아래에 남아 있고 "
                  "메뉴 6) 으로 재개할 수 있다.")
        except Exception as exc:  # 메뉴가 예외로 죽지 않게 한다
            print(f"\n[오류] {type(exc).__name__}: {exc}")
            print("  설정 이름/경로와 형식을 확인하라. 자세한 내용은 README_KO.md 참조.")
        print(MENU)


# ============================================================================
# 섹션: entrypoint  —  자체 점검과 진입점
# ============================================================================
def selftest() -> int:
    """내장 설정이 원래 JSON 과 같은지 확인한다. **시뮬레이션을 실행하지 않는다.**

    `configs/*.json` 이 이 파일 옆에 있으면 해석된 설정의 sha256 을 비교하고,
    없으면 내장 설정이 스키마 검증을 통과하는지만 확인한다.
    """
    cfg_dir = PROJECT_ROOT / "configs"
    n_ok = n_diff = n_skip = 0
    print("내장 설정 자체 점검 (스키마 검증 + JSON 대조). 시뮬레이션은 실행하지 않는다.")
    for name in sorted(BUILTIN_CONFIGS):
        try:
            builtin = resolve_config(BUILTIN_CONFIGS[name]())
        except ConfigError as exc:
            print(f"  [실패] {name}: 스키마 검증 실패 — {exc}")
            n_diff += 1
            continue
        h_builtin = config_hash(builtin)
        path = cfg_dir / f"{name}.json"
        if not path.is_file():
            print(f"  [건너뜀] {name}: 스키마 OK, 비교할 {path.name} 이 없다 "
                  f"(sha256 {h_builtin[:16]}…)")
            n_skip += 1
            continue
        ref = load_config(path)
        ref["meta"].pop("source_path", None)
        builtin["meta"].pop("source_path", None)
        h_ref = config_hash(ref)
        if h_builtin == h_ref:
            print(f"  [일치] {name}: sha256 {h_builtin[:16]}…")
            n_ok += 1
        else:
            print(f"  [다름] {name}: 내장 {h_builtin[:16]}… vs JSON {h_ref[:16]}…")
            for key in sorted(set(builtin) | set(ref)):
                if builtin.get(key) != ref.get(key):
                    print(f"      다른 항목: {key}")
            n_diff += 1
    print(f"\n일치 {n_ok} / 다름 {n_diff} / 건너뜀 {n_skip}")
    print("이 점검은 설정 스키마만 본다. 시뮬레이션·학습은 실행하지 않았다.")
    print("과학적 수치 실험을 돌리려면 run-all 또는 --execute 명령을 쓰라.")
    return 0 if n_diff == 0 else 1


_CLI_COMMANDS = {"inspect-config", "validate", "simulate", "experiment",
                 "resume", "report", "explain-neuron", "figures", "list-runs",
                 "run-all"}


def entry(argv: list[str] | None = None) -> int:
    """인자가 없으면 한국어 메뉴, 있으면 CLI 로 넘긴다.

        python cortex_all_in_one.py                       -> 메뉴
        python cortex_all_in_one.py selftest              -> 내장 설정 점검
        python cortex_all_in_one.py validate --config minimal --execute
        python cortex_all_in_one.py run-all --config minimal --out D:/결과폴더
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return menu_main()
    if args[0] == "selftest":
        return selftest()
    if args[0] in ("-h", "--help") or args[0] in _CLI_COMMANDS:
        return cli_main(args)
    print(f"[오류] 알 수 없는 명령: {args[0]!r}")
    print(f"  사용 가능한 명령: selftest, {', '.join(sorted(_CLI_COMMANDS))}")
    print("  인자 없이 실행하면 한국어 메뉴가 뜬다.")
    return 2


if __name__ == "__main__":
    raise SystemExit(entry())
