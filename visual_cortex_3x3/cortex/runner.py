"""runner.py -- 모델 조립과 실험 실행기.

명세 10, 11, 14절. **메뉴(run.py)와 CLI 는 모두 이 모듈을 호출한다.**
시뮬레이션 구현을 두 벌 만들지 않는다.

``--execute`` 없이 호출하면 아무 것도 실행하지 않고 규모·설정만 계산해서
돌려준다 (dry-run). 실제 런타임은 측정 전에 확정하지 않는다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import anatomy as anatomy_mod
from . import areas as areas_mod
from . import plasticity as plasticity_mod
from . import rng as rng_mod
from . import stimuli as stimuli_mod
from .config import (SizeEstimate, config_hash, drive_headroom_warnings,
                     estimate_sizes)
from .dynamics import Engine, ExternalDrive, rates_to_drive
from .events import CapacityExceeded, EventLog
from .recording import (
    STATUS_COMPLETED,
    STATUS_INTERRUPTED,
    CheckpointManager,
    RecordingError,
    RunRecorder,
)
from .retina import InputDriver, RetinaEncoder
from .retinotopy import LogPolarSampler
from .synapses import SynapseTable


@dataclass
class Model:
    """조립된 모델 묶음."""

    cfg: dict[str, Any]
    anat: anatomy_mod.Anatomy
    table: SynapseTable
    event_log: EventLog
    engine: Engine
    plasticity: Any
    encoder: RetinaEncoder
    sampler: LogPolarSampler
    driver: InputDriver
    wiring_report: areas_mod.WiringReport
    rngs: rng_mod.RngStreams
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return len(self.anat.population)


def build_model(cfg: dict[str, Any], rngs: rng_mod.RngStreams) -> Model:
    """설정에서 해부 구조·배선·엔진을 조립한다 (시뮬레이션은 하지 않는다)."""
    anat = anatomy_mod.build(cfg, rngs.get("wiring"))
    table, wreport = areas_mod.build(cfg, anat, rngs.get("wiring"), rngs.get("weights"))
    rec = cfg["recording"]
    log = EventLog(
        n_neurons=len(anat.population), mode=rec["mode"],
        selected_neurons=rec["selected_neurons"] or None,
        selection_criterion=rec["selection_criterion"],
        max_events=int(rec["max_events"]),
    )
    anat.population.attach(event_log=log)
    plast = plasticity_mod.make(cfg, table, anat.population.arrays, anat.ids)
    engine = Engine(cfg, anat, table, log, plasticity=plast)
    encoder = RetinaEncoder(cfg)
    sampler = LogPolarSampler(anat.grid, cfg)
    driver = InputDriver(cfg)
    return Model(cfg=cfg, anat=anat, table=table, event_log=log, engine=engine,
                 plasticity=plast, encoder=encoder, sampler=sampler, driver=driver,
                 wiring_report=wreport, rngs=rngs,
                 meta={"config_sha256": config_hash(cfg)})


def select_recording_neurons(cfg: dict[str, Any], model: Model) -> np.ndarray:
    """상태를 기록할 뉴런 선택.

    ``full`` 모드에서도 상태 저장은 용량이 크므로, 영역마다 고르게 뽑되
    **선택 기준을 manifest 에 남긴다** (조용히 버리지 않는다).
    """
    rec = cfg["recording"]
    if rec["selected_neurons"]:
        return np.asarray(rec["selected_neurons"], dtype=np.int64)
    a = model.anat.population.arrays
    ids = model.anat.ids
    picks: list[np.ndarray] = []
    per_area = max(1, int(rec["max_state_samples"] //
                          max(1, len(ids.areas) * 200)))
    wanted = rec["selected_areas"] or ids.areas.names()
    for aname in wanted:
        m = np.nonzero(a.area_id == ids.areas.id_of(aname))[0]
        if m.size == 0:
            continue
        step = max(1, m.size // per_area)
        picks.append(m[::step][:per_area])
    return np.concatenate(picks) if picks else np.arange(min(32, a.n), dtype=np.int64)


# ----------------------------------------------------------------------
def transmission_headroom(cfg: dict[str, Any], model: "Model") -> dict[str, Any]:
    """배선 규칙마다 **시냅스 전달이 임계에 닿을 수 있는지** 손계산한다.

    ``conductance_lif`` 모드의 정상상태 근사::

        g_need = gL * (V_th - EL) / (E_rev - V_th)      [nS]
        g(R)   = deg * w * (tau/1000) * R               [nS]
        R_need = g_need / (deg * w * tau/1000)          [Hz]

    ``deg`` 는 이 규칙이 표적 뉴런 1개에 만든 평균 시냅스 수, ``w`` 는 평균
    가중치다. 앞 영역이 낼 수 있는 최대 발화율은 불응기로 막히므로
    ``1000 / t_ref_ms`` 를 상한으로 쓴다. ``R_need`` 가 그 상한을 넘으면 그
    단계는 **어떤 입력에도 전달되지 않는다** — 실행은 오류 없이 끝나고 결과만
    조용히 비게 되므로 여기서 미리 표시한다.

    흥분성 규칙만 본다 (억제는 임계를 넘길 일이 없다). 단일 구획 정상상태
    근사이므로 정확한 예측이 아니라 **자릿수 점검**이다.
    """
    if cfg["engine"]["mode"] != "conductance_lif":
        return {"applicable": False,
                "reason_ko": "conductance_lif 모드에서만 계산한다."}
    a = model.anat.population.arrays
    t = model.table
    ids = model.anat.ids
    rows: list[dict[str, Any]] = []
    blocked: list[str] = []
    for r_idx, rule in enumerate(cfg["wiring"]["rules"]):
        if not rule.get("enabled", True):
            continue
        rec = cfg["receptors"][rule["receptor"]]
        if rec["kind"] != "excitatory":
            continue
        sel = t.rule_index == r_idx
        n = int(sel.sum())
        if n == 0:
            continue
        dst = t.dst_id[sel].astype(np.int64)
        src = t.src_id[sel].astype(np.int64)
        targets = np.unique(dst)
        deg = n / max(1, targets.size)
        w = float(t.weight[sel].mean())
        comp = int(np.bincount(t.target_compartment[sel].astype(np.int64)).argmax())
        gL = float(a.gL_nS[targets, comp].mean())
        EL = float(a.EL_mV[targets, comp].mean())
        V_th = float(a.threshold[targets].mean())
        E_rev = float(rec["E_rev_mV"])
        tau_s = float(rec["tau_ms"]) / 1000.0
        driving = E_rev - V_th
        if driving <= 0.0 or deg <= 0.0 or w <= 0.0:
            continue
        g_need = gL * (V_th - EL) / driving
        rate_need = g_need / (deg * w * tau_s)
        src_u = np.unique(src)
        t_ref = float(np.mean(a.t_ref_ms[src_u]))
        rate_max = 1000.0 / t_ref if t_ref > 0 else float("inf")
        # 망막은 불응기가 아니라 외부 구동이 상한을 정한다. 실제로 낼 수 없는
        # 발화율을 근거로 "닿는다" 고 적으면 진단이 무의미해진다.
        src_area_name = ids.areas.name_of(int(a.area_id[src_u[0]]))
        if cfg["anatomy"]["areas"][src_area_name]["kind"] == "retina":
            drive = cfg["retina"]["drive"]
            rate_max = min(rate_max,
                           float(drive["baseline_rate_hz"])
                           + float(drive["gain"]) * float(drive["max_rate_hz"]))
        ok = rate_need <= rate_max
        rows.append({
            "rule": rule["name"], "n_synapses": n,
            "mean_in_degree": round(deg, 3), "mean_weight_nS": round(w, 4),
            "g_need_nS": round(g_need, 4),
            "presyn_rate_needed_hz": round(rate_need, 2),
            "presyn_rate_max_hz": round(rate_max, 2),
            "presyn_area": src_area_name,
            "reachable": bool(ok),
        })
        if not ok:
            blocked.append(rule["name"])
    return {
        "applicable": True, "rules": rows, "blocked_rules": blocked,
        "note_ko": ("단일 구획 정상상태 근사다. 정확한 예측이 아니라 자릿수 "
                    "점검이며, reachable=false 인 단계는 앞 영역이 최대 속도로 "
                    "발화해도 표적을 임계까지 올리지 못한다는 뜻이다."),
    }


def silent_area_report(results: Sequence["SampleResult"]) -> dict[str, Any]:
    """모든 표본에서 한 번도 발화하지 않은 영역을 찾는다.

    앞 영역이 발화했는데 뒤 영역이 전부 침묵했다면 그 사이 전달이 끊긴 것이다.
    "스파이크 0건" 은 오류 없이 끝나므로 결과를 읽는 사람이 모형의 결론으로
    오해하기 쉽다. 여기서 명시적으로 남긴다.
    """
    totals: dict[str, int] = {}
    for r in results:
        for area, n in r.spikes_by_area.items():
            totals[area] = totals.get(area, 0) + int(n)
    silent = sorted(k for k, v in totals.items() if v == 0)
    active = sorted(k for k, v in totals.items() if v > 0)
    return {
        "spikes_by_area_total": totals,
        "silent_areas": silent,
        "active_areas": active,
        "all_silent": bool(totals) and not active,
        "note_ko": ("침묵한 영역이 있으면 시냅스 전달이 임계에 닿는지 "
                    "manifest 의 transmission_headroom 을 보라. 가중치·발화율 "
                    "상한이 모자라면 그 단계는 어떤 입력에도 반응하지 않는다."),
    }


# ----------------------------------------------------------------------
@dataclass
class SampleResult:
    sample_index: int
    stimulus_id: str
    label: str
    n_spikes_total: int
    spikes_by_area: dict[str, int]
    mean_rate_hz_by_area: dict[str, float]
    first_spike_step_by_area: dict[str, int]
    silent_fraction_by_area: dict[str, float]
    n_steps: int
    n_events: int
    settle: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class ExperimentRunner:
    """실험 실행기.

    ``execute=False`` (기본) 면 규모 추정만 하고 **아무 것도 실행하지 않는다**.
    """

    def __init__(self, cfg: dict[str, Any], run_dir: Path | None, command: str,
                 package_root: Path, execute: bool = False,
                 progress: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir) if run_dir else None
        self.command = command
        self.package_root = Path(package_root)
        self.execute = bool(execute)
        self.progress = progress or (lambda msg: None)

    # ------------------------------------------------------------------
    def inspect(self) -> dict[str, Any]:
        """설정과 규모만 계산한다. 시뮬레이션·데이터 생성 없음."""
        est: SizeEstimate = estimate_sizes(self.cfg)
        lim = self.cfg["experiment"]["limits"]
        warnings: list[str] = []
        if est.disk_mb_estimated > lim["max_disk_mb"]:
            warnings.append(
                f"추정 디스크 사용량 {est.disk_mb_estimated:.0f} MB 가 한도 "
                f"{lim['max_disk_mb']} MB 를 넘는다. 실행은 한도에서 중단된다.")
        if est.ram_mb_estimated > lim["max_ram_mb"]:
            warnings.append(
                f"추정 RAM {est.ram_mb_estimated:.0f} MB 가 한도 "
                f"{lim['max_ram_mb']} MB 를 넘는다.")
        warnings.extend(drive_headroom_warnings(self.cfg))
        return {
            "config_name": self.cfg["meta"]["name"],
            "config_sha256": config_hash(self.cfg),
            "engine_mode": self.cfg["engine"]["mode"],
            "learning_mode": self.cfg["learning"]["mode"],
            "recording_mode": self.cfg["recording"]["mode"],
            "estimate": est.to_dict(),
            "warnings": warnings,
            "experiment_status": "not_run",
            "note_ko": ("이 출력은 실행 전 추정이다. 실제 런타임은 측정하기 전에 "
                        "확정하지 않는다. 실행하려면 --execute 를 붙여라."),
        }

    # ------------------------------------------------------------------
    def _prepare(self, recorder: RunRecorder) -> Model:
        for msg in drive_headroom_warnings(self.cfg):
            recorder.warn(msg)
        rngs = rng_mod.from_config(self.cfg)
        t0 = time.time()
        self.progress("모델을 조립하는 중...")
        model = build_model(self.cfg, rngs)
        recorder.log(f"모델 조립 완료: 뉴런 {model.n_neurons}, "
                     f"시냅스 {model.table.n_synapses} ({time.time() - t0:.1f}s)")
        recorder.manifest["model"] = {
            "n_neurons": model.n_neurons,
            "n_synapses": model.table.n_synapses,
            "wiring": model.wiring_report.to_dict(),
            "anatomy": model.anat.meta,
            "synapse_summary": model.table.summary(),
        }
        recorder.set_selected_neurons(select_recording_neurons(self.cfg, model))
        recorder.manifest["recording"]["selection_criterion"] = (
            self.cfg["recording"]["selection_criterion"]
            or "영역마다 균등 간격으로 뽑은 표본 (recording.selected_neurons 미지정)")
        head = transmission_headroom(self.cfg, model)
        recorder.manifest["transmission_headroom"] = head
        for name in head.get("blocked_rules", []):
            recorder.warn(
                f"배선 규칙 {name!r} 은 앞 영역이 최대 속도로 발화해도 표적을 "
                f"임계까지 올리지 못한다 (manifest 의 transmission_headroom 참조). "
                f"이 단계 뒤쪽 영역은 어떤 입력에도 침묵할 수 있다.")
        recorder._write_manifest()
        model.engine.recorder = recorder
        return model

    def _run_sample(self, model: Model, recorder: RunRecorder, stim: Any,
                    sample_index: int, learn: bool) -> SampleResult:
        cfg = self.cfg
        a = model.anat.population.arrays
        ids = model.anat.ids
        rs = cfg["engine"]["reset_between_samples"]
        if not cfg["engine"]["sequence_mode"]:
            model.engine.reset_transient(
                voltages=rs["voltages"], conductances=rs["conductances"],
                event_queue=rs["event_queue"], traces=rs["traces"])
        model.engine.sample_id = sample_index

        n_steps = int(round(cfg["engine"]["duration_ms"] / cfg["engine"]["dt_ms"]))
        frames = stim.frames
        steps_per_frame = max(1, n_steps // len(frames))

        spikes_before = a.spike_count.copy()
        first_spike_step: dict[str, int] = {}
        saved_plast = model.engine.plasticity
        if not learn:
            model.engine.plasticity = None
        try:
            for fi, frame in enumerate(frames):
                out = model.encoder.encode(frame)
                values, smeta = model.sampler.sample(out.channels)
                values = model.encoder.normalize(values)
                rates = model.driver.rates_hz(values)          # (C, S)
                nid = model.anat.retina_neuron_id
                mask = nid >= 0
                drive = rates_to_drive(
                    cfg, nid[mask].ravel(), rates[mask].ravel(), steps_per_frame,
                    model.rngs.get("input_noise", sample_index))
                model.engine.set_external_drive(drive)
                if drive.n_multi_event_truncated:
                    recorder.warn(
                        "Poisson 모드에서 한 스텝에 2건 이상 발생한 사건을 1 스파이크로 "
                        "잘랐다. dt 를 줄이거나 최대 발화율을 낮춰라.",
                        n_truncated=int(drive.n_multi_event_truncated),
                        sample_index=sample_index, frame=fi)
                for _ in range(steps_per_frame):
                    rep = model.engine.step()
                    if rep.n_spikes:
                        fired = np.nonzero(a.fired)[0]
                        for aid in np.unique(a.area_id[fired]):
                            an = ids.areas.name_of(int(aid))
                            first_spike_step.setdefault(an, rep.step)
        finally:
            model.engine.plasticity = saved_plast

        delta = a.spike_count - spikes_before
        by_area: dict[str, int] = {}
        rate_by_area: dict[str, float] = {}
        silent_by_area: dict[str, float] = {}
        dur_s = cfg["engine"]["duration_ms"] / 1000.0
        for aname in ids.areas.names():
            m = a.area_id == ids.areas.id_of(aname)
            if not m.any():
                continue
            by_area[aname] = int(delta[m].sum())
            rate_by_area[aname] = float(delta[m].mean() / max(dur_s, 1e-9))
            silent_by_area[aname] = float(np.count_nonzero(delta[m] == 0) / m.sum())

        if model.plasticity is not None and getattr(model.plasticity, "records", None):
            recorder.record_updates(model.plasticity.records)
            model.plasticity.records = []

        return SampleResult(
            sample_index=sample_index, stimulus_id=stim.stimulus_id, label=stim.label,
            n_spikes_total=int(delta.sum()), spikes_by_area=by_area,
            mean_rate_hz_by_area=rate_by_area,
            first_spike_step_by_area=first_spike_step,
            silent_fraction_by_area=silent_by_area,
            n_steps=n_steps, n_events=model.event_log.n_events,
        )

    # ------------------------------------------------------------------
    def simulate(self) -> dict[str, Any]:
        """자극을 순서대로 제시하고 기록한다 (학습은 설정에 따름)."""
        if not self.execute:
            return self.inspect()
        if self.run_dir is None:
            raise ValueError("run_dir 가 필요하다")
        with RunRecorder(self.run_dir, self.cfg, self.command,
                         self.package_root) as recorder:
            model = self._prepare(recorder)
            ckpt = CheckpointManager(recorder, self.cfg["checkpoint"]["keep_last"])
            rng_stim = model.rngs.get("stimulus")
            stims = self._apply_stimulus_cap(
                stimuli_mod.generate(self.cfg, rng_stim), recorder)
            if not stims:
                raise ValueError("experiment.stimuli 가 비어 있어 제시할 자극이 없다")
            recorder.log(f"자극 {len(stims)}개 제시 예정")

            fit_imgs = [s.frames[0] for s in stims[:min(len(stims), 16)]]
            norm = model.encoder.fit_normalization(fit_imgs, source="simulate_prefix",
                                                   sampler=model.sampler)
            recorder.manifest["input_normalization"] = norm
            recorder.manifest["input_normalization"]["note_ko"] = (
                "simulate 명령은 제시 자극 앞부분으로 정규화 계수를 추정한다. "
                "train/dev/test 분할 실험에서는 train 만 사용한다 (experiment 명령).")
            recorder._write_manifest()

            learn = self.cfg["learning"]["mode"] == "stdp_homeostasis"
            results: list[SampleResult] = []
            try:
                for i, st in enumerate(stims):
                    self.progress(f"자극 {i + 1}/{len(stims)} ({st.stimulus_id}) — "
                                  f"중단하려면 Ctrl+C")
                    r = self._run_sample(model, recorder, st, i, learn)
                    results.append(r)
                    recorder.metric(kind="sample", learning=self.cfg["learning"]["mode"],
                                    engine_mode=self.cfg["engine"]["mode"], **r.to_dict())
                    if (self.cfg["checkpoint"]["enabled"]
                            and (i + 1) % max(1, self.cfg["checkpoint"]["every_samples"]) == 0):
                        ckpt.save(f"sample_{i:05d}",
                                  engine_state=model.engine.state_dict(),
                                  rng_state=model.rngs.state_dict(), sample_index=i,
                                  event_log_position=model.event_log.n_events,
                                  data_locator={"stimulus_id": st.stimulus_id,
                                                "stimulus_index": i})
            except CapacityExceeded as exc:
                recorder.error("용량 한도 초과로 중단", exc)
                ckpt.save("capacity_stop", engine_state=model.engine.state_dict(),
                          rng_state=model.rngs.state_dict(),
                          sample_index=len(results),
                          event_log_position=model.event_log.n_events,
                          data_locator={"reason": "capacity"})
                recorder.record_events(model.event_log)
                recorder.finish(STATUS_INTERRUPTED, str(exc))
                return {"status": STATUS_INTERRUPTED, "reason": str(exc),
                        "run_dir": str(self.run_dir)}

            recorder.record_events(model.event_log)
            self._write_aggregates(recorder, model)
            recorder.manifest["event_log_schema"] = model.event_log.schema()
            recorder.manifest["plasticity_summary"] = model.plasticity.summary()
            silent = silent_area_report(results)
            recorder.manifest["silent_areas"] = silent
            for area in silent["silent_areas"]:
                recorder.warn(f"영역 {area!r} 이 모든 표본에서 한 번도 발화하지 "
                              f"않았다. 결과를 모형의 결론으로 읽지 말 것.")
            recorder._write_manifest()
            return {"status": STATUS_COMPLETED, "run_dir": str(self.run_dir),
                    "n_samples": len(results),
                    "n_events": model.event_log.n_events,
                    "silent_areas": silent["silent_areas"]}

    # ------------------------------------------------------------------
    def experiment(self) -> dict[str, Any]:
        """train/dev/test 분할 실험. 정규화는 train 만 사용한다."""
        if not self.execute:
            info = self.inspect()
            info["protocol"] = self.cfg["experiment"]["protocol"]
            return info
        if self.run_dir is None:
            raise ValueError("run_dir 가 필요하다")
        with RunRecorder(self.run_dir, self.cfg, self.command,
                         self.package_root) as recorder:
            model = self._prepare(recorder)
            ckpt = CheckpointManager(recorder, self.cfg["checkpoint"]["keep_last"])
            stims = self._apply_stimulus_cap(
                stimuli_mod.generate(self.cfg, model.rngs.get("stimulus")), recorder)
            splits = stimuli_mod.split_stimuli(stims, self.cfg, model.rngs.get("split"))
            report = stimuli_mod.split_report(splits)
            recorder.manifest["splits"] = report
            if not report["no_overlap"]:
                raise ValueError(f"분할이 겹친다: {report['overlaps']}")

            norm = model.encoder.fit_normalization(
                [s.frames[0] for s in splits["train"]], source="train",
                sampler=model.sampler)
            recorder.manifest["input_normalization"] = norm
            recorder.manifest["test_access"] = {
                "n_test_evaluations": 0,
                "note_ko": ("test 는 선택 완료 후 조건·시드당 한 번만 평가한다. "
                            "파일럿 중 test 생성/평가 횟수를 여기에 기록한다."),
            }
            recorder._write_manifest()

            learn_splits = {"train": self.cfg["learning"]["mode"] == "stdp_homeostasis",
                            "dev": False, "test": False}
            features: dict[str, list[np.ndarray]] = {}
            labels: dict[str, list[str]] = {}
            all_results: list[SampleResult] = []
            idx = 0
            for split in ("train", "dev", "test"):
                feats: list[np.ndarray] = []
                labs: list[str] = []
                for j, st in enumerate(splits[split]):
                    self.progress(f"[{split}] {j + 1}/{len(splits[split])} "
                                  f"({st.stimulus_id}) — 중단하려면 Ctrl+C")
                    before = model.anat.population.arrays.spike_count.copy()
                    r = self._run_sample(model, recorder, st, idx, learn_splits[split])
                    idx += 1
                    all_results.append(r)
                    recorder.metric(kind="sample", split=split, **r.to_dict())
                    feats.append(self._readout_features(model, before))
                    labs.append(st.label)
                    if (self.cfg["checkpoint"]["enabled"]
                            and (j + 1) % max(1, self.cfg["checkpoint"]["every_samples"]) == 0):
                        ckpt.save(f"{split}_{j:05d}",
                                  engine_state=model.engine.state_dict(),
                                  rng_state=model.rngs.state_dict(), sample_index=idx,
                                  event_log_position=model.event_log.n_events,
                                  data_locator={"split": split, "index": j})
                features[split] = feats
                labels[split] = labs
                if split == "test":
                    recorder.manifest["test_access"]["n_test_evaluations"] += 1

            if self.cfg["readout"]["enabled"]:
                from .analysis import train_readout
                res = train_readout(self.cfg, features, labels,
                                    model.rngs.get("readout"))
                recorder.metric(kind="readout", **res)
                recorder.manifest["readout_result"] = res
            recorder.record_events(model.event_log)
            self._write_aggregates(recorder, model)
            recorder.manifest["plasticity_summary"] = model.plasticity.summary()
            silent = silent_area_report(all_results)
            recorder.manifest["silent_areas"] = silent
            for area in silent["silent_areas"]:
                recorder.warn(f"영역 {area!r} 이 모든 표본에서 한 번도 발화하지 "
                              f"않았다. 결과를 모형의 결론으로 읽지 말 것.")
            recorder._write_manifest()
            return {"status": STATUS_COMPLETED, "run_dir": str(self.run_dir),
                    "splits": report["counts"],
                    "silent_areas": silent["silent_areas"]}

    def _apply_stimulus_cap(self, stims: list[Any], recorder: RunRecorder) -> list[Any]:
        """``experiment.max_stimuli`` 로 자극 수를 자른다 (0 이면 그대로).

        자른 사실은 manifest 와 경고 로그에 남긴다. 조용히 줄이지 않는다.
        """
        cap = int(self.cfg["experiment"]["max_stimuli"])
        if cap <= 0 or len(stims) <= cap:
            recorder.manifest["stimulus_cap"] = {"cap": cap, "n_generated": len(stims),
                                                 "n_used": len(stims), "truncated": False}
            recorder._write_manifest()
            return list(stims)
        recorder.warn("experiment.max_stimuli 로 자극 목록을 앞에서부터 잘랐다.",
                      n_generated=len(stims), n_used=cap)
        recorder.manifest["stimulus_cap"] = {"cap": cap, "n_generated": len(stims),
                                             "n_used": cap, "truncated": True}
        recorder._write_manifest()
        return list(stims[:cap])

    def _readout_features(self, model: Model, spikes_before: np.ndarray) -> np.ndarray:
        """IT(또는 지정 영역) 집단 활동을 특징 벡터로 만든다.

        한 뉴런이나 3차원 물리 위치가 곧 카테고리 의미라고 가정하지 않는다.
        분류기는 :func:`cortex.analysis.train_readout` 의 **별도 모듈**이다.
        """
        cfg = self.cfg["readout"]
        a = model.anat.population.arrays
        ids = model.anat.ids
        if not ids.areas.has(cfg["source_area"]):
            return np.zeros(0, dtype=np.float64)
        m = a.area_id == ids.areas.id_of(cfg["source_area"])
        if cfg["source_layer"]:
            lm = np.zeros_like(m)
            for l in cfg["source_layer"]:
                lm |= a.layer_id == ids.layers.id_of(l)
            m &= lm
        return (a.spike_count - spikes_before)[m].astype(np.float64)

    def _write_aggregates(self, recorder: RunRecorder, model: Model) -> None:
        a = model.anat.population.arrays
        ids = model.anat.ids
        names, counts, rates = [], [], []
        for aname in ids.areas.names():
            for lname in ids.layers.names():
                m = ((a.area_id == ids.areas.id_of(aname))
                     & (a.layer_id == ids.layers.id_of(lname)))
                if not m.any():
                    continue
                names.append(f"{aname}/{lname}")
                counts.append(int(a.spike_count[m].sum()))
                rates.append(float(a.rate_estimate_hz[m].mean()))
        if names:
            recorder.record_layer_aggregate("layer_totals", {
                "name": np.array(names, dtype="S64"),
                "spike_count": np.asarray(counts, dtype=np.int64),
                "mean_rate_hz": np.asarray(rates, dtype=np.float64),
            })

    # ------------------------------------------------------------------
    def resume(self, checkpoint: Path | None = None,
               allow_mismatch: bool = False) -> dict[str, Any]:
        """중단된 실행을 체크포인트에서 재개한다.

        코드/설정 호환성을 검사하고, 이벤트 로그 위치를 복원해 **중복 기록을
        막는다**.
        """
        if self.run_dir is None:
            raise ValueError("run_dir 가 필요하다")
        cks = CheckpointManager.list_checkpoints(self.run_dir)
        if not cks:
            raise RecordingError(
                f"재개할 체크포인트가 없다: {self.run_dir}/checkpoints/")
        path = Path(checkpoint) if checkpoint else cks[-1]
        if not self.execute:
            return {"would_resume_from": str(path), "n_checkpoints": len(cks),
                    "experiment_status": "not_run",
                    "note_ko": "실행하려면 --execute 를 붙여라."}

        with RunRecorder(self.run_dir, self.cfg, self.command, self.package_root,
                         resume=True) as recorder:
            model = self._prepare(recorder)
            state, meta = CheckpointManager.load(
                path, expected_config_sha=recorder.manifest["config_sha256"],
                expected_code_sha=recorder.manifest["code"]["combined_sha256"],
                strict=not allow_mismatch)
            model.engine.load_state_dict(state)
            model.rngs.load_state_dict(meta["rng_state"])
            recorder.log(f"체크포인트에서 재개: {path.name} "
                         f"(sample_index={meta['sample_index']})")
            recorder.manifest["resumed_from"] = {
                "checkpoint": str(path), "sample_index": meta["sample_index"],
                "event_log_position": meta["event_log_position"],
                "allow_mismatch": bool(allow_mismatch),
            }
            recorder._write_manifest()
            stims = self._apply_stimulus_cap(
                stimuli_mod.generate(self.cfg, model.rngs.get("stimulus")), recorder)
            start = int(meta["sample_index"]) + 1
            model.encoder.fit_normalization(
                [s.frames[0] for s in stims[:min(len(stims), 16)]],
                source="resume", sampler=model.sampler)
            learn = self.cfg["learning"]["mode"] == "stdp_homeostasis"
            for i in range(start, len(stims)):
                self.progress(f"재개 {i + 1}/{len(stims)} — 중단하려면 Ctrl+C")
                r = self._run_sample(model, recorder, stims[i], i, learn)
                recorder.metric(kind="sample", resumed=True, **r.to_dict())
            recorder.record_events(model.event_log)
            return {"status": STATUS_COMPLETED, "run_dir": str(self.run_dir),
                    "resumed_from": str(path), "n_remaining": len(stims) - start}

    # ------------------------------------------------------------------
    def validate(self) -> dict[str, Any]:
        """구조적 필수 검사를 실행한다 (:mod:`cortex.validation`)."""
        from .validation import run_all

        if not self.execute:
            info = self.inspect()
            info["would_run_checks"] = True
            return info
        if self.run_dir is None:
            raise ValueError("run_dir 가 필요하다")
        with RunRecorder(self.run_dir, self.cfg, self.command,
                         self.package_root) as recorder:
            result = run_all(self.cfg, recorder, self.package_root,
                             progress=self.progress)
            recorder.metric(kind="validation", **{
                k: v for k, v in result.items() if k != "checks"})
            (self.run_dir / "validation.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=1, default=str),
                encoding="utf-8")
            if result["n_failed"] > 0 and self.cfg["validation"]["stop_experiment_on_failure"]:
                recorder.error(
                    f"구조적 필수 검사 {result['n_failed']}건 실패. 의존 실험을 중지한다.")
                recorder.finish("failed", "validation failed")
            return result


def new_run_dir(root: Path, tag: str) -> Path:
    """``runs/<tag>_<UTC타임스탬프>`` 경로를 만든다 (Windows 파일명 제약 준수)."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in tag)[:40]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(root) / f"{safe}_{stamp}"


__all__ = ["Model", "build_model", "ExperimentRunner", "SampleResult",
           "select_recording_neurons", "new_run_dir"]
