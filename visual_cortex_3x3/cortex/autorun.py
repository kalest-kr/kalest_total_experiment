"""autorun.py -- 한 번의 명령으로 전체 과정을 실행하고 **지정한 폴더**에 결과를 쓴다.

메뉴나 대화형 입력 없이 다음 단계를 순서대로 자동 수행한다.

===================  ==========================================================
단계                 내용
===================  ==========================================================
``config_check``     설정 해석·검증, 규모 추정 (실행 없음)
``validate``         필수 검증 1~14
``simulate``         자극 제시 시뮬레이션
``experiment``       train/dev/test 분할 실험 + readout
``reference``        Rao 참조 모델, 고정 Gabor 대조, explain-neuron, 소거 재실행
``report``           저장된 기록으로 한국어 보고서 생성
``figures``          저장된 기록으로 그림 생성
===================  ==========================================================

결과는 ``<출력폴더>/<설정이름>/<단계>/`` 아래에 쌓이고, 최상위에
``summary.json``, ``SUMMARY_ko.md``, ``run_all.log`` 가 생긴다.

**실행 정책**: 이 명령은 사용자가 출력 폴더를 명시해 직접 부를 때만 동작한다.
``--dry-run`` 으로 계획과 규모만 볼 수 있다. 한 단계가 실패해도 나머지는 계속
진행하고 단계별 상태를 기록한다. 다만 ``validation.stop_experiment_on_failure``
가 참이고 ``validate`` 가 실패하면 **의존 실험(simulate/experiment)을 중지**한다
(명세 14절).

이미 ``completed`` 로 끝난 결과 폴더는 **덮어쓰지 않는다**. ``--overwrite`` 를
주면 기존 폴더를 ``<이름>_old_<UTC>`` 로 옮겨 두고 새로 만든다.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import config as config_mod
from . import rng as rng_mod
from . import stimuli as stimuli_mod
from . import validation as validation_mod
from .analysis import (
    ablation_rerun,
    explain_neuron,
    make_report,
    read_events,
)
from .dynamics import rates_to_drive
from .events import EVENT_TYPE_INDEX
from .predictive_coding import RaoModel
from .recording import RunRecorder, library_versions
from .runner import ExperimentRunner, build_model, new_run_dir
from .v1_reference import (
    FixedGaborReference,
    contrast_reversal_response,
    orientation_tuning,
    phase_sweep_response,
)
from .visualization import make_all, plot_rao

ALL_STAGES: tuple[str, ...] = (
    "config_check", "validate", "simulate", "experiment",
    "reference", "report", "figures",
)

STAGE_TITLES_KO: dict[str, str] = {
    "config_check": "설정 해석·검증과 규모 추정",
    "validate": "필수 검증 1~14",
    "simulate": "자극 제시 시뮬레이션",
    "experiment": "train/dev/test 분할 실험",
    "reference": "Rao 참조 모델·고정 Gabor·뉴런 조회·소거 재실행",
    "report": "한국어 보고서 생성",
    "figures": "그림 생성",
}


@dataclass
class StageResult:
    """단계 하나의 결과."""

    name: str
    status: str = "pending"        # completed | failed | skipped | dry_run
    elapsed_sec: float = 0.0
    output_dir: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name, "title_ko": STAGE_TITLES_KO.get(self.name, ""),
            "status": self.status, "elapsed_sec": round(self.elapsed_sec, 3),
            "output_dir": self.output_dir, "detail": self.detail,
            "error": self.error,
        }


def resolve_backend(cfg: dict[str, Any], choice: str) -> tuple[str, str]:
    """기록 백엔드를 정한다. ``auto`` 는 h5py 가 있으면 hdf5, 없으면 npz.

    바꾼 경우 그 사실을 문자열로 돌려주어 manifest·요약·터미널에 남긴다.
    조용히 바꾸지 않는다.
    """
    if choice != "auto":
        return choice, ""
    wanted = cfg["recording"]["backend"]
    if wanted != "hdf5":
        return wanted, ""
    try:
        import h5py  # noqa: F401,PLC0415
    except ImportError:
        return "npz", (
            "h5py 가 없어 기록 백엔드를 hdf5 -> npz 로 바꿨다. "
            "npz 는 메모리에 모았다가 종료 시 저장하므로 대규모 실행에는 맞지 않는다. "
            "`python -m pip install h5py` 후 --backend hdf5 로 다시 실행하면 된다."
        )
    return "hdf5", ""


def prepare_config(name_or_path: str, *, backend: str = "auto",
                   max_stimuli: int = 0, duration_ms: float | None = None,
                   seed: int | None = None,
                   loader: Callable[[str], dict[str, Any]] | None = None
                   ) -> tuple[dict[str, Any], list[str]]:
    """설정을 읽고 자동 실행용 덮어쓰기를 적용한다. 바꾼 항목을 함께 돌려준다."""
    cfg = (loader or config_mod.load)(name_or_path)
    notes: list[str] = []
    chosen, note = resolve_backend(cfg, backend)
    if chosen != cfg["recording"]["backend"]:
        cfg["recording"]["backend"] = chosen
    if note:
        notes.append(note)
    if max_stimuli and max_stimuli > 0:
        cfg["experiment"]["max_stimuli"] = int(max_stimuli)
        notes.append(f"자극 수를 앞에서부터 {max_stimuli}개로 제한했다 (--limit-stimuli).")
    if duration_ms is not None:
        notes.append(f"engine.duration_ms 를 {cfg['engine']['duration_ms']} -> "
                     f"{duration_ms} 로 바꿨다 (--duration-ms).")
        cfg["engine"]["duration_ms"] = float(duration_ms)
    if seed is not None:
        notes.append(f"seeds.master 를 {cfg['seeds']['master']} -> {seed} 로 "
                     f"바꿨다 (--seed).")
        cfg["seeds"]["master"] = int(seed)
    config_mod.validate(cfg)
    return cfg, notes


#: 단계별 "끝났다" 표시 파일. ``manifest.json`` 이 없는 단계(RunRecorder 를 쓰지
#: 않는 단계)도 결과를 덮어쓰지 않도록 각자의 산출물 이름을 적어 둔다.
_DONE_MARKERS: dict[str, tuple[str, ...]] = {
    "config_check": ("inspect.json",),
    "reference": ("reference.json",),
}


def _prepare_dir(path: Path, overwrite: bool,
                 done_files: Sequence[str] = ()) -> str | None:
    """결과 폴더를 준비한다. 완료된 기록이 있으면 덮어쓰지 않는다."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return None
    manifest = path / "manifest.json"
    done = any((path / name).is_file() for name in done_files)
    if not done and manifest.is_file():
        try:
            done = json.loads(manifest.read_text(encoding="utf-8")).get(
                "status") == "completed"
        except json.JSONDecodeError:
            done = False
    if not done:
        return None
    if not overwrite:
        raise FileExistsError(
            f"이미 완료된 결과가 있다: {path}\n"
            f"  결과를 덮어쓰지 않는다. --overwrite 를 주거나 다른 --out 을 쓰라."
        )
    moved = path.with_name(f"{path.name}_old_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    shutil.move(str(path), str(moved))
    path.mkdir(parents=True, exist_ok=True)
    return str(moved)


# ----------------------------------------------------------------------
def stage_config_check(cfg: dict[str, Any], out: Path, package_root: Path,
                       command: str) -> dict[str, Any]:
    runner = ExperimentRunner(cfg, None, command, package_root, execute=False)
    info = runner.inspect()
    out.mkdir(parents=True, exist_ok=True)
    (out / "inspect.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return info


def stage_validate(cfg: dict[str, Any], out: Path, package_root: Path,
                   command: str, progress: Callable[[str], None]) -> dict[str, Any]:
    with RunRecorder(out, cfg, command, package_root) as recorder:
        result = validation_mod.run_all(cfg, recorder, package_root,
                                        progress=progress)
        recorder.metric(kind="validation",
                        **{k: v for k, v in result.items() if k != "checks"})
        (out / "validation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        if result["n_failed"]:
            recorder.finish("failed", f"{result['n_failed']}건 실패")
    return {k: v for k, v in result.items() if k != "checks"}


def stage_simulate(cfg: dict[str, Any], out: Path, package_root: Path,
                   command: str, progress: Callable[[str], None]) -> dict[str, Any]:
    runner = ExperimentRunner(cfg, out, command, package_root, execute=True,
                              progress=progress)
    return runner.simulate()


def stage_experiment(cfg: dict[str, Any], out: Path, package_root: Path,
                     command: str, progress: Callable[[str], None]) -> dict[str, Any]:
    runner = ExperimentRunner(cfg, out, command, package_root, execute=True,
                              progress=progress)
    return runner.experiment()


def stage_reference(cfg: dict[str, Any], out: Path, sim_dir: Path | None,
                    progress: Callable[[str], None]) -> dict[str, Any]:
    """Rao 참조 모델·고정 Gabor 대조·뉴런 조회·소거 재실행.

    이 네 가지는 simulate/experiment 경로에서 저절로 실행되지 않으므로 여기서
    따로 돌린다. **서로 다른 엔진의 결과를 섞어 순위를 매기지 않는다.**
    """
    out.mkdir(parents=True, exist_ok=True)
    detail: dict[str, Any] = {}
    rngs = rng_mod.from_config(cfg)
    model = build_model(cfg, rngs)
    stims = stimuli_mod.generate(cfg, rngs.get("stimulus"))
    cap = int(cfg["experiment"]["max_stimuli"])
    if cap > 0:
        stims = stims[:cap]
    if not stims:
        return {"skipped": True, "reason_ko": "자극이 없다"}
    model.encoder.fit_normalization([s.frames[0] for s in stims],
                                    source="reference", sampler=model.sampler)

    def sampled(img: np.ndarray) -> np.ndarray:
        values, _ = model.sampler.sample(model.encoder.encode(img).channels)
        return model.encoder.normalize(values)

    # --- (1) Rao 참조 모델 -------------------------------------------
    progress("Rao 참조 모델 정착/학습")
    rao_cfg = cfg["learning"]["rao"]
    vecs = np.stack([sampled(s.frames[0]).ravel() for s in stims[:8]])
    rao = RaoModel(n_modules=1, input_dim=int(vecs.shape[1]),
                   n1=int(rao_cfg["level_sizes"][0]),
                   n2=int(rao_cfg["level_sizes"][1] if len(rao_cfg["level_sizes"]) > 1
                          else rao_cfg["level_sizes"][0]),
                   sigma=float(rao_cfg["sigma"]), sigma_td=float(rao_cfg["sigma_td"]),
                   alpha=float(rao_cfg["alpha"]), lam=float(rao_cfg["lambda_u"]),
                   rng=rngs.get("diagnostics"))
    history: list[dict[str, Any]] = []
    state = None
    I = vecs[:1]
    for i, v in enumerate(vecs):
        I = v[None, :]
        state = rao.settle(I, steps=int(rao_cfg["settle_steps"]),
                           r_step=float(rao_cfg["r_step"]))
        upd = rao.learn_step(I, state, float(rao_cfg["u_step"]))
        history.append({"sample": i, **state.errors, **upd})
    fd = rao.finite_difference_check(I, state.r1, state.r2,
                                     eps=float(cfg["validation"]["finite_difference_eps"]),
                                     n_probe=6, rng=rngs.get("diagnostics"))
    detail["rao"] = {
        "summary": rao.summary(),
        "history": history,
        "finite_difference": {k: v for k, v in fd.items() if k != "details"},
        "note_ko": ("Rao 참조 모델은 전도도 LIF 회로와 다른 엔진이다. "
                    "동일 아키텍처 대조군인 것처럼 섞어 순위를 매기지 않는다."),
    }
    try:
        detail["rao"]["figure"] = str(plot_rao(state, I, rao, out / "rao.png"))
    except Exception as exc:  # 그림 실패가 수치 결과를 버리게 하지 않는다
        detail["rao"]["figure_error"] = f"{type(exc).__name__}: {exc}"

    # --- (2) 고정 Gabor 대조 경로 -------------------------------------
    progress("고정 Gabor 대조 경로 측정")
    ref = FixedGaborReference(model.anat.grid, cfg)
    g = model.anat.grid
    order = np.argsort(g.ecc_deg)
    centers = np.stack([g.x_deg[order], g.y_deg[order]], axis=1)[::max(1, len(order) // 8)][:8]
    size = int(cfg["retina"]["image"]["max_side_px"])
    cx = cy = (size - 1) / 2.0
    n_ori = int(cfg["v1"]["n_orientations"])
    ori_imgs = [stimuli_mod.grating_patch(size, size, cx, cy, 0.3 * size,
                                          np.pi * i / n_ori, 0.0, 0.06)
                for i in range(n_ori)]
    lum = 0  # 0번 채널 = luminance_ON
    energies = []
    for img in ori_imgs:
        r = ref.responses(sampled(img)[lum], centers)
        energies.append(r["energy"])
    energy = np.stack(energies, axis=2).max(axis=2)  # (n_c, n_ori)
    tuning = orientation_tuning(
        np.stack([e[:, 0] for e in energies], axis=1), ref.orientations_rad)
    phases = [stimuli_mod.grating_patch(size, size, cx, cy, 0.3 * size, 0.0,
                                        2 * np.pi * i / 8, 0.06) for i in range(8)]
    sweep = phase_sweep_response(ref, [sampled(p)[lum] for p in phases], centers)
    rev = contrast_reversal_response(ref, sampled(ori_imgs[0])[lum],
                                     sampled(phases[4])[lum], centers)
    detail["fixed_gabor_reference"] = {
        "enabled_in_config": bool(cfg["v1"]["fixed_gabor_reference"]["enabled"]),
        "n_centers": int(centers.shape[0]),
        "orientation_tuning": tuning,
        "phase_sweep": sweep,
        "contrast_reversal": rev,
        "learned": False,
        "note_ko": ("고정 Gabor 는 사람이 계수를 정한 특징 추출기다. 이 경로의 방향 "
                    "선택성을 '학습되었다'고 보고하지 않는다. 피질 회로 경로의 결과와 "
                    "섞어 쓰지 않는다."),
    }

    # --- (3) 저장된 기록에서 뉴런 조회 --------------------------------
    if sim_dir is not None and Path(sim_dir).is_dir():
        progress("explain-neuron (저장된 기록 조회)")
        ev = read_events(Path(sim_dir))
        nid = None
        if ev:
            spike = ev["event_type"] == EVENT_TYPE_INDEX["spike"]
            if spike.any():
                ids, counts = np.unique(ev["src_id"][spike], return_counts=True)
                nid = int(ids[int(np.argmax(counts))])
        if nid is None:
            detail["explain_neuron"] = {
                "found": False,
                "reason_ko": "저장된 기록에 발화 사건이 없어 조회할 뉴런을 고르지 못했다.",
            }
        else:
            info = explain_neuron(Path(sim_dir), nid, 0.0, float("inf"), top_k=20)
            detail["explain_neuron"] = info
            (out / "explain_neuron.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=1, default=str),
                encoding="utf-8")

    # --- (4) 소거 재실행 비교 ------------------------------------------
    progress("소거(ablation) 재실행 비교")
    n_steps = max(4, int(round(cfg["engine"]["duration_ms"]
                               / cfg["engine"]["dt_ms"] / 4)))
    ids, values = model.anat.retina_neuron_id, sampled(stims[0].frames[0])
    mask = ids >= 0
    rates = model.driver.rates_hz(values)
    drive = rates_to_drive(cfg, ids[mask].ravel(), rates[mask].ravel(), n_steps,
                           rngs.get("input_noise"))
    trainable = np.nonzero(model.table.active)[0]
    pick = trainable[: min(32, trainable.size)]
    if pick.size:
        abl = ablation_rerun(cfg, lambda: build_model(cfg, rng_mod.from_config(cfg)),
                             pick, n_steps, drive, rngs)
        detail["ablation"] = abl
    else:
        detail["ablation"] = {"skipped": True, "reason_ko": "제거할 연결이 없다"}

    (out / "reference.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return detail


def stage_report(run_dirs: Sequence[Path]) -> dict[str, Any]:
    made: list[str] = []
    errors: dict[str, str] = {}
    for d in run_dirs:
        try:
            made.append(str(make_report(Path(d))))
        except Exception as exc:
            errors[str(d)] = f"{type(exc).__name__}: {exc}"
    return {"reports": made, "errors": errors}


def stage_figures(cfg: dict[str, Any], run_dirs: Sequence[Path],
                  progress: Callable[[str], None]) -> dict[str, Any]:
    made: list[str] = []
    errors: dict[str, str] = {}
    model = None
    try:
        model = build_model(cfg, rng_mod.from_config(cfg))
    except Exception as exc:
        errors["build_model"] = f"{type(exc).__name__}: {exc}"
    for d in run_dirs:
        d = Path(d)
        nids: list[int] = []
        try:
            ev = read_events(d)
            if ev:
                spike = ev["event_type"] == EVENT_TYPE_INDEX["spike"]
                if spike.any():
                    ids, counts = np.unique(ev["src_id"][spike], return_counts=True)
                    nids = [int(x) for x in ids[np.argsort(-counts)][:3]]
        except Exception as exc:
            errors[f"{d}/events"] = f"{type(exc).__name__}: {exc}"
        progress(f"그림 생성: {d.name}")
        made += [str(p) for p in make_all(d, model, nids)]
    return {"figures": made, "n_figures": len(made), "errors": errors}


# ----------------------------------------------------------------------
def run_everything(out_dir: str | Path, configs: Sequence[str], *,
                   package_root: Path, command: str,
                   backend: str = "auto", stages: Sequence[str] | None = None,
                   limit_stimuli: int = 0, duration_ms: float | None = None,
                   seed: int | None = None, dry_run: bool = False,
                   overwrite: bool = False, stop_on_fail: bool = False,
                   config_loader: Callable[[str], dict[str, Any]] | None = None,
                   progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """전체 과정을 자동 실행하고 ``out_dir`` 아래에 결과를 쓴다.

    Parameters
    ----------
    out_dir : 결과를 쓸 폴더 (없으면 만든다)
    configs : 설정 이름 또는 JSON 경로 목록
    package_root : 코드 해시 대상 (패키지 폴더 또는 단일 파일)
    command : manifest 에 남길 실행 명령 문자열
    backend : ``auto`` | ``hdf5`` | ``npz``
    stages : 실행할 단계 목록 (기본 :data:`ALL_STAGES`)
    dry_run : True 면 계획과 규모만 계산하고 아무 것도 실행하지 않는다

    Returns
    -------
    dict : 전체 요약 (``summary.json`` 과 같은 내용)
    """
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stage_list = list(stages) if stages else list(ALL_STAGES)
    unknown = [s for s in stage_list if s not in ALL_STAGES]
    if unknown:
        raise ValueError(f"알 수 없는 단계: {unknown} (가능: {list(ALL_STAGES)})")

    log_path = out_root / "run_all.log"
    log_fh = open(log_path, "a", encoding="utf-8")

    def emit(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()
        if progress is not None:
            progress(msg)

    started = time.time()
    emit(f"출력 폴더: {out_root}")
    emit(f"설정: {list(configs)} / 단계: {stage_list}"
         + (" / dry-run (아무 것도 실행하지 않는다)" if dry_run else ""))

    summary: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": command, "output_dir": str(out_root),
        "configs": list(configs), "stages": stage_list, "dry_run": bool(dry_run),
        "backend_choice": backend, "library_versions": library_versions(),
        "results": {},
    }

    overall_ok = True
    for name in configs:
        emit("=" * 70)
        emit(f"설정 '{name}' 시작")
        cfg_dir = out_root / Path(name).stem
        cfg_dir.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {"config": name, "stages": [], "notes": []}
        summary["results"][Path(name).stem] = entry
        try:
            cfg, notes = prepare_config(name, backend=backend,
                                        max_stimuli=limit_stimuli,
                                        duration_ms=duration_ms, seed=seed,
                                        loader=config_loader)
        except Exception as exc:
            emit(f"  [실패] 설정을 읽지 못했다: {exc}")
            entry["config_error"] = f"{type(exc).__name__}: {exc}"
            overall_ok = False
            continue
        entry["notes"] = notes
        entry["config_sha256"] = config_mod.config_hash(cfg)
        for n in notes:
            emit(f"  [알림] {n}")

        run_dirs: list[Path] = []
        sim_dir: Path | None = None
        validate_failed = False

        for stage in stage_list:
            res = StageResult(name=stage)
            t0 = time.time()
            # 번호는 고른 단계 순서가 아니라 ALL_STAGES 안의 고정 위치를 쓴다.
            # 일부 단계만 돌려도 폴더 이름이 전체 실행과 같아야 비교할 수 있다.
            target = cfg_dir / f"{ALL_STAGES.index(stage) + 1:02d}_{stage}"
            try:
                if dry_run:
                    if stage == "config_check":
                        res.detail = stage_config_check(cfg, target, package_root,
                                                        command)
                        res.output_dir = str(target)
                    else:
                        res.detail = {"note_ko": "dry-run 이라 실행하지 않았다."}
                    res.status = "dry_run"
                elif stage in ("simulate", "experiment") and validate_failed:
                    res.status = "skipped"
                    res.error = ("validate 가 실패했고 "
                                 "validation.stop_experiment_on_failure 가 참이라 "
                                 "의존 실험을 중지했다.")
                    emit(f"  [{stage}] 건너뜀 — {res.error}")
                else:
                    emit(f"  [{stage}] {STAGE_TITLES_KO[stage]} 시작")
                    if stage in ("config_check", "validate", "simulate",
                                 "experiment", "reference"):
                        moved = _prepare_dir(target, overwrite,
                                             _DONE_MARKERS.get(stage, ()))
                        if moved:
                            emit(f"    기존 결과를 옮겨 두었다: {moved}")
                        res.output_dir = str(target)
                    if stage == "config_check":
                        res.detail = stage_config_check(cfg, target, package_root,
                                                        command)
                    elif stage == "validate":
                        res.detail = stage_validate(cfg, target, package_root,
                                                    command, emit)
                        run_dirs.append(target)
                        if res.detail.get("n_failed"):
                            validate_failed = bool(
                                cfg["validation"]["stop_experiment_on_failure"])
                            res.status = "failed"
                            res.error = f"{res.detail['n_failed']}건 실패"
                    elif stage == "simulate":
                        res.detail = stage_simulate(cfg, target, package_root,
                                                    command, emit)
                        run_dirs.append(target)
                        sim_dir = target
                    elif stage == "experiment":
                        res.detail = stage_experiment(cfg, target, package_root,
                                                      command, emit)
                        run_dirs.append(target)
                    elif stage == "reference":
                        res.detail = stage_reference(cfg, target, sim_dir, emit)
                    elif stage == "report":
                        res.detail = stage_report(run_dirs)
                        res.output_dir = str(cfg_dir)
                    elif stage == "figures":
                        res.detail = stage_figures(cfg, run_dirs, emit)
                        res.output_dir = str(cfg_dir)
                    if res.status == "pending":
                        res.status = "completed"
            except Exception as exc:
                res.status = "failed"
                res.error = f"{type(exc).__name__}: {exc}"
                res.detail["traceback"] = traceback.format_exc()
                emit(f"  [{stage}] 실패: {res.error}")
            res.elapsed_sec = time.time() - t0
            entry["stages"].append(res.to_dict())
            if res.status == "failed":
                overall_ok = False
                if stop_on_fail:
                    emit("  --stop-on-fail 이라 여기서 중단한다.")
                    break
            elif res.status in ("completed", "dry_run"):
                emit(f"  [{stage}] {res.status} ({res.elapsed_sec:.1f}s)")
        entry["run_dirs"] = [str(d) for d in run_dirs]

    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["elapsed_sec"] = round(time.time() - started, 3)
    summary["all_stages_ok"] = overall_ok
    summary["experiment_status"] = "not_run" if dry_run else (
        "completed" if overall_ok else "failed")
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    md = write_summary_markdown(out_root, summary)
    emit("=" * 70)
    emit(f"끝. 요약: {out_root / 'summary.json'}")
    emit(f"      한국어 요약: {md}")
    emit(f"      로그: {log_path}")
    log_fh.close()
    return summary


def write_summary_markdown(out_root: Path, summary: dict[str, Any]) -> Path:
    """``SUMMARY_ko.md`` 를 요약 dict 에서만 만든다 (손으로 넣은 수치 없음)."""
    L: list[str] = []
    A = L.append
    A("# 자동 실행 요약")
    A("")
    A(f"- 출력 폴더: `{summary['output_dir']}`")
    A(f"- 실행 명령: `{summary['command']}`")
    A(f"- 시작(UTC): {summary['started_utc']} / 종료(UTC): {summary.get('finished_utc')}")
    A(f"- 소요(초): {summary.get('elapsed_sec')}")
    A(f"- dry-run: {summary['dry_run']}")
    A(f"- 전체 상태: **{summary.get('experiment_status')}**")
    A("")
    A("> 이 문서는 `summary.json` 에서만 생성되었다. 손으로 넣은 성능 숫자는 없다.")
    A("")
    for cfg_name, entry in summary["results"].items():
        A(f"## 설정 `{cfg_name}`")
        A("")
        if entry.get("config_error"):
            A(f"- 설정 오류: {entry['config_error']}")
            A("")
            continue
        A(f"- 설정 해시: `{entry.get('config_sha256', '')[:32]}…`")
        for n in entry.get("notes", []):
            A(f"- 알림: {n}")
        A("")
        A("| 단계 | 내용 | 상태 | 소요(초) | 출력 |")
        A("|---|---|---|---|---|")
        for st in entry["stages"]:
            out = st.get("output_dir") or ""
            if out:
                out = f"`{Path(out).name}`"
            A(f"| {st['stage']} | {st['title_ko']} | **{st['status']}** | "
              f"{st['elapsed_sec']} | {out} |")
        A("")
        for st in entry["stages"]:
            d = st.get("detail") or {}
            if st["stage"] == "config_check" and d.get("estimate"):
                e = d["estimate"]
                A(f"- 규모 추정: 뉴런 {e['n_neurons']}, 시냅스 {e['n_synapses_estimated']}, "
                  f"스텝 {e['n_steps']}, 이벤트 {e['events_estimated']} "
                  f"(런타임은 측정 전에 확정하지 않는다)")
            if st["stage"] == "validate" and "n_passed" in d:
                A(f"- 필수 검증: 통과 {d['n_passed']} / 실패 {d['n_failed']} / "
                  f"건너뜀 {d['n_skipped']} (skipped 는 passed 에 포함하지 않는다)")
            if st["stage"] == "simulate" and d.get("n_samples") is not None:
                A(f"- 시뮬레이션: 표본 {d.get('n_samples')}개, "
                  f"이벤트 {d.get('n_events')}건")
            if st["stage"] == "experiment" and d.get("splits"):
                A(f"- 분할: {d['splits']}")
            if st["stage"] in ("simulate", "experiment") and d.get("silent_areas"):
                A(f"- **경고**: {st['stage']} 단계에서 영역 "
                  f"{', '.join(d['silent_areas'])} 이(가) 모든 표본에서 한 번도 "
                  f"발화하지 않았다. 이 영역의 0 은 모형의 결론이 아니라 전달이 "
                  f"끊겼다는 표시다. 해당 실행 폴더의 `manifest.json` 에서 "
                  f"`transmission_headroom` 의 `blocked_rules` 를 보라.")
            if st["stage"] == "reference":
                rao = (d.get("rao") or {}).get("finite_difference")
                if rao:
                    A(f"- Rao 유한차분 상대 잔차: {rao.get('max_relative_residual')}")
                fg = (d.get("fixed_gabor_reference") or {}).get("orientation_tuning")
                if fg:
                    A(f"- 고정 Gabor 대조 경로 OSI 평균: {fg.get('osi_mean')} "
                      f"(학습된 선택성이 아니다)")
                abl = d.get("ablation") or {}
                if "n_neurons_changed" in abl:
                    A(f"- 소거 재실행: 연결 {abl['n_ablated_synapses']}개 제거 시 "
                      f"발화가 바뀐 뉴런 {abl['n_neurons_changed']}개")
            if st["stage"] == "figures" and d.get("n_figures") is not None:
                A(f"- 그림 {d['n_figures']}개")
            if st.get("error"):
                A(f"- 오류: {st['error']}")
        A("")
    A("## 해석 시 주의")
    A("")
    A("- 이 실행의 수치는 **모형 파라미터로 만든 모형의 출력**이다. 생물학적 측정값이 아니다.")
    A("- 고정 Gabor 대조 경로의 선택성은 학습된 것이 아니다.")
    A("- Rao 참조 모델은 전도도 LIF 회로와 다른 엔진이므로 섞어 순위를 매기지 않는다.")
    A("- 정지영상만 쓴 실행의 운동 지표는 `not_applicable` 이다.")
    A("- 자세한 가정은 BIOLOGY_AND_ASSUMPTIONS.md 를 보라.")
    path = out_root / "SUMMARY_ko.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


__all__ = [
    "ALL_STAGES", "STAGE_TITLES_KO", "StageResult", "resolve_backend",
    "prepare_config", "run_everything", "write_summary_markdown",
    "stage_config_check", "stage_validate", "stage_simulate",
    "stage_experiment", "stage_reference", "stage_report", "stage_figures",
]
