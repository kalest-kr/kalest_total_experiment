"""recording.py -- 실행 기록기, 저장소 백엔드, 체크포인트 관리.

명세 10절. 사용자가 명령을 실행하면 ``runs/<run_id>/`` 아래에 다음이 생긴다.

==================  ==================================================
파일                내용
==================  ==================================================
``manifest.json``   run_id, UTC 시각, 실행 명령, 해석된 전체 설정, 코드/데이터
                    해시, 라이브러리 버전, 장치/dtype/스레드, 종·모형 가정,
                    단위, 기록 모드, status
``events.h5``       입력·도착·발화·학습 사건 (+ 인덱스)
``states.h5``       선택 뉴런의 soma/basal/apical 상태, 전도도, 임계값, 활동률,
                    층별 집계
``updates.h5``      시냅스/임계 변화, 규칙별 항, 제한·정규화 전후 변화량
``metrics.jsonl``   실제 실행된 스텝/에폭의 측정치와 측정 조건
``summary.csv``     metrics 의 표 형태 요약
``run.log``         진행 로그
``errors.jsonl``    경고/예외/중단 원인/재개 정보
``checkpoints/``    가중치, 상태, 흔적, 이벤트 큐, RNG 상태, 데이터 위치,
                    로그 처리 위치
``report.md``       저장된 결과로만 생성하는 한국어 보고서 (analysis 모듈)
==================  ==================================================

status 는 ``running / completed / interrupted / failed`` 중 하나다.
**이미 ``completed`` 인 실행 디렉터리는 덮어쓰지 않는다.**

h5py 는 필수 의존성이지만 **지연 import** 한다. 설치되어 있지 않으면 한국어로
설치 방법을 알리고, ``recording.backend = "npz"`` 로 바꿀 수 있음을 안내한다.
npz 백엔드는 메모리에 모았다가 종료 시 저장하므로 대규모 실행에는 적합하지 않다
(이 한계를 manifest 에 기록한다).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_INTERRUPTED = "interrupted"
STATUS_FAILED = "failed"


class RecordingError(RuntimeError):
    pass


def _require_h5py():
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:
        raise RecordingError(
            "h5py 가 설치되어 있지 않다.\n"
            "  설치: python -m pip install -r requirements.txt\n"
            "  또는 설정에서 recording.backend 를 \"npz\" 로 바꿔라 "
            "(npz 는 메모리에 모았다가 종료 시 저장하므로 대규모 실행에는 맞지 않는다)."
        ) from exc
    return h5py


# ----------------------------------------------------------------------
class ArrayStore:
    """추가 가능한 열 저장소의 공통 인터페이스."""

    def append(self, group: str, columns: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def write_once(self, group: str, columns: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class Hdf5Store(ArrayStore):
    """압축 청크 HDF5 저장소."""

    def __init__(self, path: Path, compression: str = "gzip", level: int = 4,
                 chunk_rows: int = 4096) -> None:
        h5py = _require_h5py()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self.path, "a")
        self.compression = compression
        self.level = int(level)
        self.chunk_rows = int(chunk_rows)

    def append(self, group: str, columns: dict[str, np.ndarray]) -> None:
        g = self._f.require_group(group)
        for name, arr in columns.items():
            a = np.asarray(arr)
            if name not in g:
                maxshape = (None,) + a.shape[1:]
                chunks = (min(self.chunk_rows, max(1, a.shape[0])),) + a.shape[1:]
                g.create_dataset(name, data=a, maxshape=maxshape, chunks=chunks,
                                 compression=self.compression,
                                 compression_opts=self.level)
            else:
                ds = g[name]
                n0 = ds.shape[0]
                ds.resize(n0 + a.shape[0], axis=0)
                ds[n0:] = a

    def write_once(self, group: str, columns: dict[str, np.ndarray]) -> None:
        g = self._f.require_group(group)
        for name, arr in columns.items():
            if name in g:
                del g[name]
            g.create_dataset(name, data=np.asarray(arr),
                             compression=self.compression,
                             compression_opts=self.level)

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:  # pragma: no cover
            pass


class NpzStore(ArrayStore):
    """메모리 누적 후 종료 시 .npz 로 저장하는 대체 백엔드.

    대규모 실행에는 적합하지 않다. 이 한계는 manifest 에 기록된다.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path).with_suffix(".npz")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buf: dict[str, list[np.ndarray]] = {}
        self._once: dict[str, np.ndarray] = {}

    def append(self, group: str, columns: dict[str, np.ndarray]) -> None:
        for name, arr in columns.items():
            self._buf.setdefault(f"{group}/{name}", []).append(np.asarray(arr))

    def write_once(self, group: str, columns: dict[str, np.ndarray]) -> None:
        for name, arr in columns.items():
            self._once[f"{group}/{name}"] = np.asarray(arr)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        out: dict[str, np.ndarray] = dict(self._once)
        for k, parts in self._buf.items():
            if parts:
                out[k] = np.concatenate(parts)
        np.savez_compressed(self.path, **out)


def make_store(path: Path, cfg: dict[str, Any]) -> ArrayStore:
    rec = cfg["recording"]
    if rec["backend"] == "npz":
        return NpzStore(path)
    return Hdf5Store(path, rec["compression"], rec["compression_level"],
                     rec["chunk_rows"])


# ----------------------------------------------------------------------
def code_hash(package_dir: Path) -> dict[str, Any]:
    """패키지 소스 파일들의 sha256 목록과 전체 해시."""
    files = sorted(p for p in package_dir.rglob("*.py")
                   if "__pycache__" not in p.parts)
    per: dict[str, str] = {}
    h = hashlib.sha256()
    for p in files:
        d = hashlib.sha256(p.read_bytes()).hexdigest()
        per[str(p.relative_to(package_dir.parent))] = d
        h.update(d.encode())
    return {"combined_sha256": h.hexdigest(), "files": per, "n_files": len(files)}


def git_commit(root: Path) -> str | None:
    """git 커밋 해시. git 이 없거나 저장소가 아니면 None (네트워크 접근 없음)."""
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() or None
    except Exception:
        return None


def library_versions() -> dict[str, str]:
    vers: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "matplotlib", "PIL", "h5py", "numba"):
        try:
            mod = __import__(name)
            vers[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            vers[name] = "not_installed"
    return vers


def thread_settings() -> dict[str, str]:
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS")
    return {k: os.environ.get(k, "unset") for k in keys}


def data_hash(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        if p.is_file():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ----------------------------------------------------------------------
@dataclass
class RunPaths:
    root: Path
    manifest: Path
    events: Path
    states: Path
    updates: Path
    metrics: Path
    summary_csv: Path
    log: Path
    errors: Path
    checkpoints: Path
    figures: Path
    report: Path

    @staticmethod
    def make(run_dir: Path) -> "RunPaths":
        r = Path(run_dir)
        return RunPaths(
            root=r, manifest=r / "manifest.json", events=r / "events.h5",
            states=r / "states.h5", updates=r / "updates.h5",
            metrics=r / "metrics.jsonl", summary_csv=r / "summary.csv",
            log=r / "run.log", errors=r / "errors.jsonl",
            checkpoints=r / "checkpoints", figures=r / "figures",
            report=r / "report.md",
        )


class RunRecorder:
    """실행 기록기. 계산 결과를 바꾸지 않는다.

    Parameters
    ----------
    run_dir : Path
    cfg : 해석된 설정
    command : str
        실제로 실행한 명령 문자열.
    package_root : Path
        소스 해시 계산용 패키지 경로.
    """

    def __init__(self, run_dir: Path, cfg: dict[str, Any], command: str,
                 package_root: Path, data_files: list[Path] | None = None,
                 resume: bool = False) -> None:
        self.paths = RunPaths.make(Path(run_dir))
        self.cfg = cfg
        self.command = command
        self.package_root = Path(package_root)
        self.resume = bool(resume)

        if self.paths.manifest.is_file():
            old = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
            if old.get("status") == STATUS_COMPLETED and not resume:
                raise RecordingError(
                    f"이미 완료된 실행 디렉터리다: {self.paths.root}\n"
                    f"  결과를 덮어쓰지 않는다. 다른 --run-dir 를 쓰거나 "
                    f"기존 결과를 다른 곳으로 옮겨라."
                )
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoints.mkdir(parents=True, exist_ok=True)
        self.paths.figures.mkdir(parents=True, exist_ok=True)

        self.run_id = self.paths.root.name
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._t0 = time.time()
        self.status = STATUS_RUNNING
        self._events_store: ArrayStore | None = None
        self._states_store: ArrayStore | None = None
        self._updates_store: ArrayStore | None = None
        self._metrics_fh = open(self.paths.metrics, "a", encoding="utf-8")
        self._log_fh = open(self.paths.log, "a", encoding="utf-8")
        self._err_fh = open(self.paths.errors, "a", encoding="utf-8")
        self._metric_rows: list[dict[str, Any]] = []
        self._state_rows = 0
        self._selected: np.ndarray | None = None
        self._step_every = int(cfg["recording"]["state_sample_every_steps"])
        self._flush_every = int(cfg["recording"]["flush_every_steps"])
        self._max_state_rows = int(cfg["recording"]["max_state_samples"])
        self.n_state_rows_skipped = 0

        self.manifest: dict[str, Any] = self._build_manifest(data_files or [])
        self._write_manifest()

    # ------------------------------------------------------------------
    def _build_manifest(self, data_files: list[Path]) -> dict[str, Any]:
        cfgm = self.cfg
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_utc": self.started_utc,
            "finished_utc": None,
            "elapsed_seconds": None,
            "command": self.command,
            "resumed": self.resume,
            "config": cfgm,
            "config_sha256": _sha256_json(cfgm),
            "code": code_hash(self.package_root),
            "git_commit": git_commit(self.package_root.parent),
            "data_hashes": data_hash(data_files),
            "library_versions": library_versions(),
            "platform": {
                "platform": platform.platform(), "machine": platform.machine(),
                "processor": platform.processor(), "python": sys.version,
            },
            "device": "cpu",
            "gpu_used": False,
            "dtypes": {"state": "float64", "weight": "float64",
                       "event_time": "float64", "ids": "int64/int32"},
            "thread_settings": thread_settings(),
            "units": _units_block(),
            "recording": {
                "mode": cfgm["recording"]["mode"],
                "backend": cfgm["recording"]["backend"],
                "selection_criterion": cfgm["recording"]["selection_criterion"],
                "state_sample_every_steps": self._step_every,
                "max_events": cfgm["recording"]["max_events"],
                "max_state_samples": self._max_state_rows,
                "npz_backend_limitation_ko": (
                    "npz 백엔드는 메모리에 모았다가 종료 시 저장한다. 대규모 실행에는 "
                    "적합하지 않다."
                ) if cfgm["recording"]["backend"] == "npz" else None,
            },
            "assumptions": {
                "species_assumption": cfgm["meta"]["species_assumption"],
                "note_ko": (
                    "영역별 뉴런 수, 층 두께, 연결 확률은 출처가 명시되지 않은 경우 "
                    "모형 파라미터다. BIOLOGY_AND_ASSUMPTIONS.md 참조. "
                    "이 결과를 인간 뇌의 완전하고 정확한 복제라고 부르지 않는다."
                ),
            },
            "experiment_status": "not_run",
        }

    def _write_manifest(self) -> None:
        self.paths.manifest.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=1, default=_jsonable),
            encoding="utf-8")

    # ------------------------------------------------------------------
    def set_selected_neurons(self, neuron_ids: np.ndarray) -> None:
        self._selected = np.asarray(neuron_ids, dtype=np.int64)
        self.manifest["recording"]["n_selected_neurons"] = int(self._selected.size)
        self._write_manifest()

    @property
    def events_store(self) -> ArrayStore:
        if self._events_store is None:
            self._events_store = make_store(self.paths.events, self.cfg)
        return self._events_store

    @property
    def states_store(self) -> ArrayStore:
        if self._states_store is None:
            self._states_store = make_store(self.paths.states, self.cfg)
        return self._states_store

    @property
    def updates_store(self) -> ArrayStore:
        if self._updates_store is None:
            self._updates_store = make_store(self.paths.updates, self.cfg)
        return self._updates_store

    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        self._log_fh.write(line + "\n")
        self._log_fh.flush()

    def warn(self, message: str, **extra: Any) -> None:
        self._err_fh.write(json.dumps(
            {"level": "warning", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "message": message, **extra}, ensure_ascii=False, default=_jsonable) + "\n")
        self._err_fh.flush()

    def error(self, message: str, exc: BaseException | None = None, **extra: Any) -> None:
        payload = {
            "level": "error",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": message, **extra,
        }
        if exc is not None:
            payload["exception_type"] = type(exc).__name__
            payload["traceback"] = "".join(traceback.format_exception(
                type(exc), exc, exc.__traceback__))
        self._err_fh.write(json.dumps(payload, ensure_ascii=False,
                                      default=_jsonable) + "\n")
        self._err_fh.flush()

    def metric(self, **fields: Any) -> None:
        """metrics.jsonl 에 한 줄 기록한다. **측정 조건도 함께 넣어라.**"""
        row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
        self._metrics_fh.write(json.dumps(row, ensure_ascii=False,
                                          default=_jsonable) + "\n")
        self._metrics_fh.flush()
        self._metric_rows.append(row)

    # ------------------------------------------------------------------
    def record_step(self, engine: Any, report: Any) -> None:
        """엔진이 매 스텝 끝에 호출한다. **계산을 바꾸지 않는다.**"""
        if self.cfg["recording"]["mode"] == "summary":
            return
        if report.step % self._step_every != 0:
            return
        if self._selected is None or self._selected.size == 0:
            return
        if self._state_rows >= self._max_state_rows:
            self.n_state_rows_skipped += int(self._selected.size)
            return
        a = engine.a
        sel = self._selected
        n = sel.size
        self.states_store.append("states", {
            "step": np.full(n, report.step, dtype=np.int64),
            "time_ms": np.full(n, report.time_ms, dtype=np.float64),
            "sample_id": np.full(n, engine.sample_id, dtype=np.int32),
            "neuron_id": sel.astype(np.int32),
            "V_soma_mV": a.V_mV[sel, 0].copy(),
            "V_basal_mV": a.V_mV[sel, 1].copy(),
            "V_apical_mV": a.V_mV[sel, 2].copy(),
            "g_total_nS": a.g_nS[sel].sum(axis=(1, 2)),
            "threshold": a.threshold[sel].copy(),
            "rate_hz": a.rate_estimate_hz[sel].copy(),
            "fired": a.fired[sel].astype(np.int8),
            "interval_activation": a.interval_activation[sel].copy(),
        })
        self._state_rows += int(n)
        if report.step % self._flush_every == 0:
            self.flush()

    def record_updates(self, records: list[Any]) -> None:
        """가소성 갱신 기록을 updates 저장소에 넣는다."""
        if not records:
            return
        cols: dict[str, list[Any]] = {}
        for r in records:
            d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
            for k, v in d.items():
                if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
                    cols.setdefault(k, []).append(v)
        if cols:
            self.updates_store.append(
                "updates", {k: np.asarray(v) for k, v in cols.items()})

    def record_events(self, event_log: Any) -> None:
        """이벤트 로그 전체를 저장소에 쓴다 (실행 종료 또는 표본 경계에서)."""
        cols = event_log.all_columns()
        if not cols or len(next(iter(cols.values()))) == 0:
            return
        self.events_store.write_once("events", cols)
        self.events_store.write_once("schema", {
            "columns": np.array(event_log.COLUMNS, dtype="S64"),
        })

    def record_layer_aggregate(self, name: str, columns: dict[str, np.ndarray]) -> None:
        self.states_store.write_once(f"aggregate/{name}", columns)

    # ------------------------------------------------------------------
    def flush(self) -> None:
        for s in (self._events_store, self._states_store, self._updates_store):
            if s is not None:
                s.flush()
        self._metrics_fh.flush()
        self._log_fh.flush()
        self._err_fh.flush()

    def write_summary_csv(self) -> None:
        if not self._metric_rows:
            return
        keys: list[str] = []
        for r in self._metric_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(self.paths.summary_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for r in self._metric_rows:
                wr.writerow({k: _flat(r.get(k)) for k in keys})

    def finish(self, status: str, note: str = "") -> None:
        """실행을 마치고 manifest 를 갱신한다 (예외 경로에서도 호출할 것)."""
        self.status = status
        self.manifest["status"] = status
        self.manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.manifest["elapsed_seconds"] = round(time.time() - self._t0, 3)
        self.manifest["finish_note"] = note
        self.manifest["n_state_rows_written"] = self._state_rows
        self.manifest["n_state_rows_skipped_by_limit"] = self.n_state_rows_skipped
        self.manifest["experiment_status"] = (
            "completed" if status == STATUS_COMPLETED else status)
        self.write_summary_csv()
        self._write_manifest()
        self.flush()
        for s in (self._events_store, self._states_store, self._updates_store):
            if s is not None:
                s.close()
        for fh in (self._metrics_fh, self._log_fh, self._err_fh):
            try:
                fh.close()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            if self.status == STATUS_RUNNING:
                self.finish(STATUS_COMPLETED)
            return False
        if issubclass(exc_type, KeyboardInterrupt):
            self.error("사용자 중단 (KeyboardInterrupt)", exc)
            self.finish(STATUS_INTERRUPTED, "KeyboardInterrupt")
        else:
            self.error("실행 중 예외 발생", exc)
            self.finish(STATUS_FAILED, f"{exc_type.__name__}: {exc}")
        return False


# ----------------------------------------------------------------------
@dataclass
class Checkpoint:
    """체크포인트 내용 (재개에 필요한 모든 것)."""

    step_index: int
    sample_index: int
    payload: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """체크포인트 저장/복원. 재개 시 호환성을 검사한다."""

    def __init__(self, recorder: RunRecorder, keep_last: int = 3) -> None:
        self.recorder = recorder
        self.dir = recorder.paths.checkpoints
        self.keep_last = int(keep_last)

    def save(self, name: str, *, engine_state: dict[str, Any],
             rng_state: dict[str, Any], sample_index: int,
             event_log_position: int, data_locator: dict[str, Any],
             extra: dict[str, Any] | None = None) -> Path:
        """체크포인트 1개를 저장한다 (npz + json 메타)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        base = self.dir / name
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, Any] = {
            "name": name, "sample_index": int(sample_index),
            "event_log_position": int(event_log_position),
            "data_locator": data_locator,
            "config_sha256": self.recorder.manifest["config_sha256"],
            "code_sha256": self.recorder.manifest["code"]["combined_sha256"],
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "extra": extra or {},
            "rng_state": rng_state,
        }
        for k, v in engine_state.items():
            if isinstance(v, np.ndarray):
                arrays[f"engine/{k}"] = v
            elif k == "queue":
                meta["queue_steps"] = [int(s) for s in v["steps"]]
                for i, payload in enumerate(v["payloads"]):
                    for pk, pv in payload.items():
                        arrays[f"queue/{i}/{pk}"] = np.asarray(pv)
            else:
                meta[f"engine_{k}"] = v
        np.savez_compressed(base.with_suffix(".npz"), **arrays)
        base.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1, default=_jsonable),
            encoding="utf-8")
        self._prune()
        return base.with_suffix(".npz")

    def _prune(self) -> None:
        cks = sorted(self.dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        for p in cks[:-self.keep_last] if len(cks) > self.keep_last else []:
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)

    @staticmethod
    def list_checkpoints(run_dir: Path) -> list[Path]:
        d = Path(run_dir) / "checkpoints"
        return sorted(d.glob("*.npz")) if d.is_dir() else []

    @staticmethod
    def load(path: Path, *, expected_config_sha: str | None = None,
             expected_code_sha: str | None = None,
             strict: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
        """체크포인트를 읽고 코드/설정 호환성을 검사한다.

        Returns
        -------
        (engine_state, meta)
        """
        p = Path(path)
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        problems: list[str] = []
        if expected_config_sha and meta["config_sha256"] != expected_config_sha:
            problems.append("설정 해시가 다르다 (config_sha256)")
        if expected_code_sha and meta["code_sha256"] != expected_code_sha:
            problems.append("코드 해시가 다르다 (code_sha256)")
        if problems and strict:
            raise RecordingError(
                "체크포인트 호환성 검사 실패: " + "; ".join(problems) +
                "\n  같은 코드/설정으로 재개하거나, 차이를 인지하고 --allow-mismatch 를 써라."
            )
        data = np.load(p, allow_pickle=False)
        engine_state: dict[str, Any] = {}
        queue: dict[str, Any] = {"steps": meta.get("queue_steps", []), "payloads": []}
        qpay: dict[int, dict[str, np.ndarray]] = {}
        for key in data.files:
            if key.startswith("engine/"):
                engine_state[key.split("/", 1)[1]] = data[key]
            elif key.startswith("queue/"):
                _, i, pk = key.split("/", 2)
                qpay.setdefault(int(i), {})[pk] = data[key]
        queue["payloads"] = [qpay[i] for i in sorted(qpay)]
        engine_state["queue"] = queue
        for k, v in meta.items():
            if k.startswith("engine_"):
                engine_state[k[len("engine_"):]] = v
        return engine_state, meta


# ----------------------------------------------------------------------
def _sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=_jsonable)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _flat(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, default=_jsonable)
    return v


def _units_block() -> dict[str, str]:
    from .units import UNIT_TABLE
    return dict(UNIT_TABLE)


def load_manifest(run_dir: Path) -> dict[str, Any]:
    p = Path(run_dir) / "manifest.json"
    if not p.is_file():
        raise RecordingError(
            f"실행 기록을 찾을 수 없다: {p}\n"
            f"  --run-dir 경로를 확인하라. 조회 명령은 새 실험을 시작하지 않는다."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    p = Path(run_dir) / "metrics.jsonl"
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]


__all__ = [
    "STATUS_RUNNING", "STATUS_COMPLETED", "STATUS_INTERRUPTED", "STATUS_FAILED",
    "RecordingError", "ArrayStore", "Hdf5Store", "NpzStore", "make_store",
    "code_hash", "git_commit", "library_versions", "thread_settings", "data_hash",
    "RunPaths", "RunRecorder", "Checkpoint", "CheckpointManager",
    "load_manifest", "read_metrics",
]
