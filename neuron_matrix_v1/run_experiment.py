#!/usr/bin/env python3
"""run_experiment.py -- 대조 실험 실행 (명세 [13], [16]).

사용법::

    python run_experiment.py --profile quick
    python run_experiment.py --profile main
    python run_experiment.py --profile quick --lr-search
    python run_experiment.py --profile main --conditions frozen,local_learning

산출물 (``results/<profile>/``)::

    run_summary.json                 조건·시드 집계와 환경/시간/해시
    raw/<condition>_seed<S>.json     조건·시드별 원자료
    raw/<condition>_seed<S>_predictions.npz  동일 예측 배열 (모든 시험 지표의 출처)
    metrics.csv                      조건·시드별 주요 지표
    models/<condition>_seed<S>.{json,npz}    모델 체크포인트
    figures/*.png                    활동 지도, 입력 사상, 진단 기여 그림
    diagnostics/*.json, *.csv        진단 기록
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from typing import Any

import numpy as np
import scipy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.connections import ConnectionTable
from src.datasets import render_scene
from src.diagnostics import (
    TraceRecorder,
    connection_role_table,
    counterfactual_check,
    measured_orientation_preference,
)
from src.engine import SimulationEngine
from src.events import InputEvent
from src.experiment import CONDITIONS, V1Experiment, build_dataset
from src.local_learning import FreeRunSnapshot
from src.metrics import paired_difference, summarize_seeds
from src.neuron import Population
from src.persistence import file_sha256, hash_manifest, load_model, save_model
from src.protocols import status_report
from src.teacher import teacher_feasibility_report
from src import visualization as viz

HERE = os.path.dirname(os.path.abspath(__file__))


def jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj)} is not JSON serializable")


def env_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "pillow": __import__("PIL").__version__,
        "torch": "미설치 (필수 의존성 아님)",
        "gpu": "미사용 (CPU 전용 실행)",
    }


# ----------------------------------------------------------------------
def run_single(config, seed, condition, epochs, lr, outdir, save_ckpt=True,
               eval_splits=("dev", "test", "test_novel")) -> tuple[dict, V1Experiment]:
    exp = V1Experiment(config, seed, condition, epochs, learning_rate=lr)
    t0 = time.time()
    exp.setup()
    pre_pred = exp.predict_split("train")["l2"]
    train_info = exp.train()
    ev = exp.evaluate(eval_splits)
    exp._eval_raw = ev
    w = exp.table.weight
    tr = exp.table.trainable
    guided = None
    if condition in ("correction_only", "local_learning", "shuffled_teacher"):
        gq = exp.predict_split_guided("dev")
        from src.metrics import evaluate_map
        guided = evaluate_map(exp.data.targets["dev"], gq)

    rec = {
        "condition": condition, "seed": seed, "epochs": epochs, "learning_rate": lr,
        "build_stats": exp.circuit.build_stats,
        "normalizer": exp.circuit.normalizer.to_dict(),
        "dataset_summaries": exp.data.summaries,
        "initial_activity_rate_train_l2": float(pre_pred.mean()),
        "initial_weight_stats": {
            "mean": float(exp.initial_weights[tr].mean()),
            "min": float(exp.initial_weights[tr].min()),
            "max": float(exp.initial_weights[tr].max()),
            "n_at_zero": int(np.count_nonzero(exp.initial_weights[tr] <= 0.0)),
        },
        "final_weight_stats": {
            "mean": float(w[tr].mean()), "min": float(w[tr].min()), "max": float(w[tr].max()),
            "l2_norm": float(np.linalg.norm(w[tr])),
            "n_at_zero": int(np.count_nonzero(w[tr] <= 0.0)),
            "n_at_max": int(np.count_nonzero(w[tr] >= config["learning"]["weight_max"])),
            "weight_version": exp.table.weight_version,
            "bitwise_unchanged_from_init": bool(np.array_equal(
                w.view(np.uint8), exp.initial_weights.view(np.uint8))),
        },
        "training": {k: v for k, v in train_info.items() if k != "teacher_permutation_used"},
        "teacher_permutation_applied": train_info.get("teacher_permutation_used") is not None,
        "evaluation": {k: {kk: vv for kk, vv in v.items() if kk != "_predictions"}
                       for k, v in ev.items()},
        "guided_dev_reference_only": guided,
        "elapsed_sec": time.time() - t0,
    }

    os.makedirs(os.path.join(outdir, "raw"), exist_ok=True)
    tag = f"{condition}_seed{seed}" + (f"_lr{lr}" if lr != config["learning"]["learning_rate"] else "")
    with open(os.path.join(outdir, "raw", f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1, default=jsonable)
    np.savez_compressed(
        os.path.join(outdir, "raw", f"{tag}_predictions.npz"),
        **{f"pred_{k}": v["_predictions"] for k, v in ev.items()},
        **{f"target_{k}": exp.data.targets[k] for k in ev},
        l2_ids=exp.l2_ids, l2_orientation=exp.circuit.l2_orientation,
        l2_radial_bin=exp.circuit.grid.radial_bin[exp.circuit.l2_point],
    )
    if save_ckpt:
        os.makedirs(os.path.join(outdir, "models"), exist_ok=True)
        save_model(os.path.join(outdir, "models", tag), exp.circuit,
                   extra={"condition": condition, "seed": seed, "epochs": epochs,
                          "learning_rate": lr})
    return rec, exp


# ----------------------------------------------------------------------
def run_diagnostics(config, exp: V1Experiment, outdir: str, seed: int) -> dict[str, Any]:
    """진단 산출물 (명세 [11], [16]). 학습 가중치를 바꾸지 않는다."""
    dd = os.path.join(outdir, "diagnostics")
    fd = os.path.join(outdir, "figures")
    os.makedirs(dd, exist_ok=True); os.makedirs(fd, exist_ok=True)
    out: dict[str, Any] = {}
    w_before = exp.table.snapshot_weights()

    # 1) A/B/C -> D 반사실 예제 (명세 [11])
    pop = Population()
    for nm in ("A", "B", "C", "D"):
        pop.declare(nm, threshold=0.5, transmission=1.0, observation_tick=1)
    pop.finalize()
    tb = ConnectionTable(4, pop.names())
    cA = tb.add(0, 3, "excitatory", 0.6, 1)
    cB = tb.add(1, 3, "excitatory", 0.4, 1)
    cC = tb.add(2, 3, "inhibitory", 0.6, 1, trainable=True)
    tb.finalize()
    eng = SimulationEngine(pop, tb, n_ticks=2, event_mode="object")
    evs = [InputEvent(1, cA, 0, 1.0), InputEvent(1, cB, 1, 0.5), InputEvent(1, cC, 2, 1.0)]
    base = eng.run_trial(None, trial_id="abcd", extra_events=evs)
    tb.restore_weights(np.array([0.6, 0.4, 0.2]), 0)
    inter = eng.run_trial(None, trial_id="abcd/int", extra_events=evs)
    tb.restore_weights(np.array([0.6, 0.4, 0.6]), 0)
    out["abcd_counterfactual"] = {
        "baseline": {"E": float(base.observed_E[3]), "I": float(base.observed_I[3]),
                     "u": float(base.observed_u[3]), "q": int(base.observed_q[3])},
        "intervened_wC_0.6_to_0.2": {
            "E": float(inter.observed_E[3]), "I": float(inter.observed_I[3]),
            "u": float(inter.observed_u[3]), "q": int(inter.observed_q[3])},
        "note_ko": ("C 의 가중치를 0.6 -> 0.2 로 바꾸면 u 가 0.2 -> 0.6 이 되어 발화가 바뀐다. "
                    "이는 해당 조작의 효과를 보여줄 뿐, C 가 생물학적으로 잘못된 연결이라는 "
                    "증명이 아니다."),
    }

    # 2) 실제 회로 진단 기록 (선택 뉴런/시행/최대 행 수)
    n_trials = int(config["diagnostics"]["record_trials"])
    dv0 = exp.data.drive["dev"]
    probe = exp.engine.run_trial((exp.circuit.l4_drive_ids, dv0[0]), trial_id="diag/probe")
    s_probe = probe.arrivals_at(exp.l2_obs_tick)
    n_in = np.bincount(exp.table.post_id, weights=(s_probe > 0).astype(np.float64),
                       minlength=exp.table.n_neurons)[exp.l2_ids]
    # 입력이 많이 도착한 L2 6개 + 입력이 전혀 없는 L2 2개를 함께 기록한다
    busy = exp.l2_ids[np.argsort(-n_in)][:6].tolist()
    quiet = exp.l2_ids[np.argsort(n_in)][:2].tolist()
    sel = sorted(set(busy + quiet))
    trial_ids = [f"diag/{i}" for i in range(n_trials)]
    rec = TraceRecorder(exp.pop, exp.table, select_neurons=sel, select_trials=trial_ids,
                        max_rows=int(config["diagnostics"]["max_rows"]))
    exp.engine.recorder = rec
    dv = exp.data.drive["dev"]
    scenes = exp.data.scenes["dev"]
    tg = exp.data.targets["dev"]
    for i in range(n_trials):
        exp.engine.run_trial((exp.circuit.l4_drive_ids, dv[i]), trial_id=f"diag/{i}")
    exp.engine.recorder = None
    csv_info = rec.write_csv(os.path.join(dd, "trace_states.csv"),
                             os.path.join(dd, "trace_contributions.csv"))
    out["trace_csv"] = csv_info

    inspections = []
    for i in range(n_trials):
        for nid in sel:
            j = int(np.nonzero(exp.l2_ids == nid)[0][0])
            info = rec.target_comparison(f"diag/{i}", nid, exp.l2_obs_tick, int(tg[i][j]))
            if info.get("found"):
                inspections.append(info)
    out["inspect_neuron_examples"] = inspections[:10]

    # 선택 뉴런 3x3 상태 + 기여 막대그래프
    figs = []
    if inspections:
        best = max(inspections,
                   key=lambda x: x["observed"]["n_arriving_connections"])
        figs.append(viz.plot_neuron_state_and_contributions(
            exp.circuit, best,
            os.path.join(fd, f"diag_neuron_state_seed{seed}.png"),
            title=(f"학습 후 · 자유 실행 · image_id={scenes[0].scene_id} · seed={seed} "
                   f"· 관찰 tick={exp.l2_obs_tick}")))

    # 3) 반사실 개입 (실제 회로, 상태 보존 검사 포함)
    drv = (exp.circuit.l4_drive_ids, dv[0])
    base0 = exp.engine.run_trial(drv, trial_id="cf/base")
    active_tr = np.nonzero(exp.table.trainable & (base0.arrivals_at(exp.l2_obs_tick) > 0))[0]
    cf = None
    if active_tr.size:
        cid = int(active_tr[0])
        cf = counterfactual_check(
            exp.engine, drv, {cid: 0.0}, exp.l2_ids,
            baseline_response=base0.observed_q[exp.l2_ids],
            weights_of_trial=exp.table.snapshot_weights(),
            weight_version_of_trial=exp.table.weight_version, trial_id="cf/real")
        out["counterfactual_real_circuit"] = {
            k: v for k, v in cf.items()
            if k not in ("baseline_q", "intervened_q", "baseline_u", "intervened_u", "delta_u")
        }
        out["counterfactual_real_circuit"]["connection"] = exp.table.row(cid)

        # 4) 연결 역할 표 (여러 선 위치·방향)
        idx = list(range(min(8, len(scenes))))
        drives = [(exp.circuit.l4_drive_ids, dv[i]) for i in idx]
        labels = [{"scene_id": scenes[i].scene_id, "shape": scenes[i].shape_type,
                   "orientation_deg": scenes[i].orientation_deg,
                   "contrast": scenes[i].contrast, "polarity": scenes[i].polarity}
                  for i in idx]
        post = int(exp.table.post_id[cid])
        out["connection_role_table"] = connection_role_table(
            exp.engine, drives, labels, cid, post, delta_weight=-1.0)
        out["connection_role_note_ko"] = (
            "여러 선 위치·방향에서 같은 연결의 기여와 개입 효과를 비교한 표다. "
            "단일 가중치에 고정된 의미가 들어 있다고 가정하지 않는다.")

    # 5) 측정된 방향 선호 vs 메타데이터
    pred_dev = exp._eval_raw["dev"]["_predictions"]
    meas = measured_orientation_preference(
        pred_dev, [s.orientation_deg for s in scenes])
    out["measured_orientation_preference"] = {
        "orientations": meas["orientations"], "n_ambiguous": meas["n_ambiguous"],
        "n_matching_metadata": int(np.count_nonzero(
            np.asarray(meas["measured_preferred_orientation"]) == exp.circuit.l2_orientation)),
        "n_l2": int(exp.l2_ids.size),
        "note_ko": "측정 방향 선호와 미리 지정한 preferred_orientation 메타데이터는 별개다.",
    }
    figs.append(viz.plot_orientation_preference(
        exp.circuit, meas, os.path.join(fd, f"orientation_preference_seed{seed}.png"),
        title=f"학습 후 · 자유 실행 · dev 전체 · seed={seed}"))

    # 6) 교사 실현 가능성 / 식별 불가능성
    l4_bits = np.zeros((dv.shape[0], exp.circuit.l4_drive_ids.size), dtype=np.int8)
    for i in range(dv.shape[0]):
        r = exp.engine.run_trial((exp.circuit.l4_drive_ids, dv[i]), trial_id=f"feas/{i}")
        l4_bits[i] = r.observed_q[exp.circuit.l4_drive_ids]
    out["teacher_feasibility_dev"] = teacher_feasibility_report(
        tg, l4_bits, [s.scene_id for s in scenes])
    out["l4_binarization"] = {
        "mean_active_l4_per_scene": float(l4_bits.sum(axis=1).mean()),
        "n_l4": int(exp.circuit.l4_drive_ids.size),
        "n_scenes_with_no_active_l4": int(np.count_nonzero(l4_bits.sum(axis=1) == 0)),
        "note_ko": "L4 의 이진화로 소실된 정보를 뒤의 교사가 복구해 준다고 가정하지 않는다.",
    }

    # 7) 뉴런별 양성 표본 수 / 미관측 채널
    pos_train = exp.data.targets["train"].sum(axis=0)
    out["per_neuron_training_positives"] = {
        "min": int(pos_train.min()), "max": int(pos_train.max()),
        "mean": float(pos_train.mean()),
        "n_neurons_with_zero_positive_train": int(np.count_nonzero(pos_train == 0)),
        "n_l2": int(exp.l2_ids.size),
        "note_ko": "훈련에서 양성 표본이 한 번도 없는 채널은 이 규칙으로 학습될 수 없다.",
    }

    out["weights_unchanged_by_diagnostics"] = bool(
        np.array_equal(exp.table.weight.view(np.uint8), w_before.view(np.uint8)))
    out["figures"] = figs
    with open(os.path.join(dd, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=jsonable)
    return out


def make_figures(config, exp: V1Experiment, exp_frozen: V1Experiment | None,
                 outdir: str, seed: int) -> list[str]:
    fd = os.path.join(outdir, "figures")
    os.makedirs(fd, exist_ok=True)
    figs = []
    scenes = exp.data.scenes["dev"]
    dv = exp.data.drive["dev"]
    tg = exp.data.targets["dev"]
    idx = next((i for i, s in enumerate(scenes) if s.shape_type == "horizontal_line"), 0)
    img = render_scene(scenes[idx])
    figs.append(viz.plot_input_and_sampling(
        exp.circuit, img, os.path.join(fd, f"input_mapping_seed{seed}.png"),
        title_extra=f"image_id={scenes[idx].scene_id} · seed={seed}"))

    pred_after = exp._eval_raw["dev"]["_predictions"][idx]
    figs.append(viz.plot_activity_maps(
        exp.circuit, pred_after, os.path.join(fd, f"activity_after_seed{seed}.png"),
        targets=tg[idx], analysis_mode=True,
        title=(f"학습 후 · 자유 실행 (교정 없음) · condition={exp.condition} · "
               f"image_id={scenes[idx].scene_id} · seed={seed} · 관찰 tick=2")))
    if exp_frozen is not None:
        pred_before = exp_frozen._eval_raw["dev"]["_predictions"][idx]
        figs.append(viz.plot_activity_maps(
            exp_frozen.circuit, pred_before,
            os.path.join(fd, f"activity_before_seed{seed}.png"),
            targets=tg[idx], analysis_mode=True,
            title=(f"학습 전 (frozen) · 자유 실행 (교정 없음) · "
                   f"image_id={scenes[idx].scene_id} · seed={seed} · 관찰 tick=2")))

    # 유도 실행 (교정 켬) 활동 지도 -- 참고용
    thr = exp.l2_thresholds
    free = exp.engine.run_trial((exp.circuit.l4_drive_ids, dv[idx]), trial_id="figfree")
    c = exp.teacher.make_corrections(free.observed_u[exp.l2_ids], tg[idx], thr,
                                     config["teacher"]["margin"])
    routed = exp.relay.route((exp.l2_ids, c))
    g = exp.engine.run_trial((exp.circuit.l4_drive_ids, dv[idx]), correction=routed,
                             trial_id="figguided")
    figs.append(viz.plot_activity_maps(
        exp.circuit, g.observed_q[exp.l2_ids],
        os.path.join(fd, f"activity_guided_seed{seed}.png"),
        targets=tg[idx], analysis_mode=True,
        title=(f"학습 후 · **유도 실행 (교정 켬)** · image_id={scenes[idx].scene_id} · "
               f"seed={seed} · 관찰 tick=2 · 최종 성능으로 보고하지 않음")))
    return figs


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["quick", "main"], default="quick")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--conditions", default=None, help="쉼표 구분. 기본은 config 의 4개 조건")
    ap.add_argument("--seeds", default=None, help="쉼표 구분 정수")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr-search", action="store_true",
                    help="사전 고정 후보 [0.01,0.05,0.1] 에서 dev balanced accuracy 로 선택")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    prof = config["profiles"][args.profile]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else prof["seeds"]
    epochs = args.epochs if args.epochs is not None else prof["epochs"]
    conditions = args.conditions.split(",") if args.conditions else config["conditions"]
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"알 수 없는 조건: {c}")
    outdir = args.out or os.path.join(HERE, "results", args.profile + ("_lrsearch" if args.lr_search else ""))
    os.makedirs(outdir, exist_ok=True)

    started = time.time()
    print("=" * 78)
    print(f"프로필={args.profile}  시드={seeds}  에폭={epochs}  조건={conditions}")
    print(f"학습률 탐색={'ON ' + str(config['lr_search']['candidates']) if args.lr_search else 'OFF (고정 %.3g)' % config['learning']['learning_rate']}")
    print(f"출력 디렉터리: {outdir}")
    print("=" * 78)

    records: list[dict[str, Any]] = []
    kept_exps: dict[str, V1Experiment] = {}
    lr_selection: dict[str, Any] = {}

    for cond in conditions:
        learns = cond in ("local_learning", "shuffled_teacher")
        if args.lr_search and learns:
            cands = config["lr_search"]["candidates"]
            dev_by_lr: dict[float, list[float]] = {}
            for lr in cands:
                vals = []
                for sd in seeds:
                    t0 = time.time()
                    rec, exp = run_single(config, sd, cond, epochs, lr, outdir,
                                          save_ckpt=True, eval_splits=("dev",))
                    b = rec["evaluation"]["dev"]["l2_overall"]["balanced_accuracy"]
                    vals.append(b)
                    print(f"  [탐색] {cond} seed={sd} lr={lr}: dev bacc={b:.4f} ({time.time()-t0:.1f}s)")
                dev_by_lr[lr] = vals
            means = {lr: float(np.mean(v)) for lr, v in dev_by_lr.items()}
            best = max(means, key=lambda lr: (means[lr], -lr))
            # 동률이면 작은 학습률
            top = max(means.values())
            best = min([lr for lr in means if abs(means[lr] - top) < 1e-12])
            lr_selection[cond] = {"candidates": cands, "mean_dev_balanced_accuracy": means,
                                  "selected_learning_rate": best,
                                  "per_seed_dev": dev_by_lr,
                                  "rule_ko": "여러 시드 평균 dev balanced accuracy 최대, 동률이면 작은 학습률"}
            print(f"  [선택] {cond}: lr={best} (평균 dev bacc {means})")
            # 보관한 선택 모델로 시험 -- 시험은 선택 완료 후 조건·시드당 1회
            for sd in seeds:
                rec, exp = run_single(config, sd, cond, epochs, best, outdir, save_ckpt=True)
                rec["lr_selected_by_dev"] = True
                records.append(rec)
                kept_exps.setdefault(f"{cond}_{sd}", exp)
        else:
            lr = config["learning"]["learning_rate"]
            for sd in seeds:
                t0 = time.time()
                rec, exp = run_single(config, sd, cond, epochs, lr, outdir)
                records.append(rec)
                kept_exps[f"{cond}_{sd}"] = exp
                b = rec["evaluation"]["test"]["l2_overall"]["balanced_accuracy"]
                f1 = rec["evaluation"]["test"]["l2_overall"]["f1"]
                print(f"  {cond:<17} seed={sd}  test bacc={b:.4f}  F1={f1 if f1 is None else round(f1,4)}"
                      f"  ({time.time()-t0:.1f}s)")

    # --- 집계 -----------------------------------------------------------
    summary: dict[str, Any] = {}
    for cond in conditions:
        rs = [r for r in records if r["condition"] == cond]
        if not rs:
            continue
        def pick(split, key):
            return [r["evaluation"][split]["l2_overall"][key] for r in rs]
        summary[cond] = {
            "seeds": [r["seed"] for r in rs],
            "learning_rate": rs[0]["learning_rate"],
            "balanced_accuracy": summarize_seeds(pick("test", "balanced_accuracy")),
            "f1": summarize_seeds(pick("test", "f1")),
            "dev_balanced_accuracy": summarize_seeds(pick("dev", "balanced_accuracy")),
            "test_novel_balanced_accuracy": summarize_seeds(
                pick("test_novel", "balanced_accuracy")),
            "accuracy": summarize_seeds(pick("test", "accuracy")),
            "false_positive_rate": summarize_seeds(pick("test", "false_positive_rate")),
            "false_negative_rate": summarize_seeds(pick("test", "false_negative_rate")),
            "blank_false_firing_rate": summarize_seeds(
                [r["evaluation"]["test"]["blank"]["blank_false_firing_rate"] for r in rs]),
            "guided_dev_balanced_accuracy_reference_only": summarize_seeds(
                [(r["guided_dev_reference_only"] or {}).get("balanced_accuracy") for r in rs]),
            "clipping_fraction": summarize_seeds(
                [r["training"].get("clipping_fraction_overall") for r in rs]),
            "weight_l2_norm": summarize_seeds([r["final_weight_stats"]["l2_norm"] for r in rs]),
            "E_mean_l2_test": summarize_seeds([r["evaluation"]["test"]["E_mean_l2"] for r in rs]),
            "I_mean_l2_test": summarize_seeds([r["evaluation"]["test"]["I_mean_l2"] for r in rs]),
            "n_never_active_test": summarize_seeds(
                [r["evaluation"]["test"]["per_neuron"]["n_never_active"] for r in rs]),
            "n_no_positive_target_test": summarize_seeds(
                [r["evaluation"]["test"]["per_neuron"]["n_no_positive_target"] for r in rs]),
            "n_undefined_test": summarize_seeds(
                [r["evaluation"]["test"]["per_neuron"]["n_undefined"] for r in rs]),
        }
    if "frozen" in summary:
        for cond in summary:
            if cond == "frozen":
                continue
            summary[cond]["paired_difference_vs_frozen_test_bacc"] = paired_difference(
                summary[cond]["balanced_accuracy"]["values"],
                summary["frozen"]["balanced_accuracy"]["values"])

    # frozen vs correction_only 최종(교정 제거) 동일성 검사
    equality = None
    if "frozen" in conditions and "correction_only" in conditions:
        eqs = []
        for sd in seeds:
            a = kept_exps.get(f"frozen_{sd}")
            b = kept_exps.get(f"correction_only_{sd}")
            if a and b:
                eqs.append(bool(np.array_equal(a._eval_raw["test"]["_predictions"],
                                               b._eval_raw["test"]["_predictions"])))
        equality = {
            "frozen_equals_correction_only_without_correction": eqs,
            "all_equal": all(eqs) if eqs else None,
            "note_ko": ("교정을 제거하고 일시 상태를 올바르게 초기화하면 두 조건의 반응은 같아야 한다. "
                        "같지 않다면 지속 상태 누출 가능성을 먼저 조사한다."),
        }

    # --- CSV ------------------------------------------------------------
    csv_path = os.path.join(outdir, "metrics.csv")
    cols = ["condition", "seed", "learning_rate", "epochs",
            "test_balanced_accuracy", "test_f1", "test_accuracy",
            "test_TP", "test_FP", "test_TN", "test_FN",
            "test_false_positive_rate", "test_false_negative_rate",
            "dev_balanced_accuracy", "test_novel_balanced_accuracy",
            "guided_dev_balanced_accuracy_reference_only",
            "blank_false_firing_rate", "clipping_fraction",
            "weight_l2_norm", "E_mean_l2_test", "I_mean_l2_test",
            "n_never_active_test", "n_no_positive_target_test", "elapsed_sec"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in records:
            t = r["evaluation"]["test"]["l2_overall"]
            wr.writerow({
                "condition": r["condition"], "seed": r["seed"],
                "learning_rate": r["learning_rate"], "epochs": r["epochs"],
                "test_balanced_accuracy": t["balanced_accuracy"], "test_f1": t["f1"],
                "test_accuracy": t["accuracy"],
                "test_TP": t["confusion"]["TP"], "test_FP": t["confusion"]["FP"],
                "test_TN": t["confusion"]["TN"], "test_FN": t["confusion"]["FN"],
                "test_false_positive_rate": t["false_positive_rate"],
                "test_false_negative_rate": t["false_negative_rate"],
                "dev_balanced_accuracy": r["evaluation"]["dev"]["l2_overall"]["balanced_accuracy"],
                "test_novel_balanced_accuracy":
                    r["evaluation"].get("test_novel", {}).get("l2_overall", {}).get("balanced_accuracy"),
                "guided_dev_balanced_accuracy_reference_only":
                    (r["guided_dev_reference_only"] or {}).get("balanced_accuracy"),
                "blank_false_firing_rate": r["evaluation"]["test"]["blank"]["blank_false_firing_rate"],
                "clipping_fraction": r["training"].get("clipping_fraction_overall"),
                "weight_l2_norm": r["final_weight_stats"]["l2_norm"],
                "E_mean_l2_test": r["evaluation"]["test"]["E_mean_l2"],
                "I_mean_l2_test": r["evaluation"]["test"]["I_mean_l2"],
                "n_never_active_test": r["evaluation"]["test"]["per_neuron"]["n_never_active"],
                "n_no_positive_target_test":
                    r["evaluation"]["test"]["per_neuron"]["n_no_positive_target"],
                "elapsed_sec": r["elapsed_sec"],
            })

    # --- 그림 / 진단 -----------------------------------------------------
    figures: list[str] = []
    diagnostics: dict[str, Any] = {}
    ref_seed = seeds[0]
    main_exp = kept_exps.get(f"local_learning_{ref_seed}")
    frozen_exp = kept_exps.get(f"frozen_{ref_seed}")
    if not args.no_figures and main_exp is not None:
        figures += make_figures(config, main_exp, frozen_exp, outdir, ref_seed)
        hists = {c: next((r["training"]["history"] for r in records
                          if r["condition"] == c and r["seed"] == ref_seed), [])
                 for c in conditions}
        hists = {k: v for k, v in hists.items() if v}
        if hists:
            figures.append(viz.plot_learning_curve(
                hists, os.path.join(outdir, "figures", f"learning_curve_seed{ref_seed}.png"),
                title=f"조건별 dev balanced accuracy 추이 (seed={ref_seed}, 교정 없는 평가)"))
        if summary:
            figures.append(viz.plot_condition_summary(
                summary, os.path.join(outdir, "figures", "condition_summary.png"),
                title=f"조건별 시험 성능 (시드 {seeds} 평균±표준편차, 교정 없는 평가)"))
    if main_exp is not None:
        diagnostics = run_diagnostics(config, main_exp, outdir, ref_seed)
        figures += diagnostics.get("figures", [])

    # --- 요약 저장 -------------------------------------------------------
    elapsed = time.time() - started
    code_files = [os.path.join(HERE, p) for p in
                  ["config.json", "run_checks.py", "run_experiment.py", "make_report.py",
                   "run_demo.py", "requirements.txt", "README_ko.md"]]
    code_files += [os.path.join(HERE, "src", f) for f in sorted(os.listdir(os.path.join(HERE, "src")))
                   if f.endswith(".py")]
    result_files = []
    for root, _, fs in os.walk(outdir):
        for fn in fs:
            result_files.append(os.path.join(root, fn))

    summary_doc = {
        "profile": args.profile,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": elapsed,
        "seeds": seeds, "epochs": epochs, "conditions": conditions,
        "learning_rate_fixed": config["learning"]["learning_rate"],
        "lr_search_enabled": bool(args.lr_search),
        "lr_selection": lr_selection,
        "environment": env_info(),
        "config": config,
        "summary_by_condition": summary,
        "frozen_vs_correction_only": equality,
        "records": records,
        "diagnostics": diagnostics,
        "figures": sorted(set(figures)),
        "implemented_scope": status_report(),
        "code_hashes": hash_manifest(code_files, HERE),
        "result_file_hashes": hash_manifest(result_files, HERE),
        "not_executed_ko": [
            "PyTorch / GPU: 설치하지 않았고 사용하지 않았다. 전부 NumPy/SciPy CPU 실행이다. "
            "필수 의존성이 아니므로 설치를 반복 시도하지 않았다.",
            "V2 / V4 / IT / L5 / L6: 구현하지 않았다. 입출력 프로토콜과 설계 설명만 제공하며 "
            "호출하면 NotImplementedError 를 낸다 (src/protocols.py).",
            "100만 화소(1000x1000) 입력은 전처리 API 로 **처리만** 했다. 그 규모의 회로를 구성하거나 "
            "학습하지 않았다. 학습은 64x64 입력 / 뉴런 1024개 / 연결 17408개 회로에서만 했다.",
            ("학습률 탐색: 이 실행에서는 실행했다." if bool(args.lr_search) else
             "학습률 탐색: 이 실행에서는 **미실행**이다 (고정 학습률 %.3g 모드). "
             "`--lr-search` 옵션으로 사전 고정 후보 [0.01,0.05,0.1] 탐색을 실행할 수 있다."
             % config["learning"]["learning_rate"]),
            "대화형 GUI 데모(run_demo.py 의 Matplotlib 슬라이더): 이 실행 환경에 디스플레이가 없어 "
            "**미실행**이다. 같은 추론 경로를 `--headless` 로 실행해 고정 장면 PNG 를 생성했다.",
            "0°/90° 외의 방향 채널, 가로선/세로선 외의 도형, 색채 경로, 확률적 입력: "
            "설정으로 확장 가능하지만 이번 실행에서는 **미실행**이다.",
            "시험 세트는 선택 완료 후 조건·시드당 1회만 평가했고 모든 시험 지표는 저장된 동일 예측 배열 "
            "(raw/*_predictions.npz) 에서 계산했다. 시험 결과를 보고 구현이나 후보를 바꾸지 않았다.",
        ],
    }
    sp = os.path.join(outdir, "run_summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary_doc, f, ensure_ascii=False, indent=1, default=jsonable)

    print("\n" + "=" * 78)
    for cond in summary:
        s = summary[cond]
        print(f"{cond:<18} test bacc {s['balanced_accuracy']['mean']:.4f} "
              f"± {s['balanced_accuracy']['std'] if s['balanced_accuracy']['std'] is not None else 0:.4f}"
              f"   F1 {s['f1']['mean']:.4f}")
    print(f"\n총 소요 {elapsed:.1f}s · 요약 {sp}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
