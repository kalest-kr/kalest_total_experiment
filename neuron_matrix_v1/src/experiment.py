"""experiment.py -- 대조 실험 실행기 (frozen / correction_only / local_learning / shuffled_teacher).

명세 [13]. 동일 데이터·연결·초기값에서 조건만 바꿔 비교한다.

* ``frozen``          : 가중치 고정, 교정 없는 추론.
* ``correction_only`` : 훈련 자극에서 활동 교정만 시연하고 가중치는 고정.
                        최종 평가는 교정 없이 한다.
* ``local_learning``  : 지역 교사 + 국소 규칙으로 연결 학습. 최종 평가는 교정 없이.
* ``shuffled_teacher``: 훈련 표본 간 목표 지도를 **고정 무작위 순열**로 바꿔
                        같은 규칙으로 학습. 평가 목표는 정상 목표를 쓴다.
                        검증·시험 목표는 섞지 않는다.

모든 조건은 같은 시드에서 같은 초기 가중치, 같은 데이터, 같은 표본 순서를 쓴다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .datasets import DatasetConfig, SceneSpec, generate_scenes, render_scene, split_summary
from .local_learning import FreeRunSnapshot, LearningConfig, LocalLearner
from .metrics import (
    baseline_metrics,
    blank_false_firing,
    evaluate_map,
    orientation_position_breakdown,
    per_neuron_metrics,
)
from .rngs import stream
from .teacher import L1Relay, L2Descriptor, LocalTeacher, teacher_feasibility_report
from .v1 import V1Circuit

CONDITIONS = ("frozen", "correction_only", "local_learning", "shuffled_teacher")


@dataclass
class PreparedData:
    scenes: dict[str, list[SceneSpec]]
    raw: dict[str, np.ndarray]        # split -> (n, 2P) 정규화 전 표본
    drive: dict[str, np.ndarray]      # split -> (n, 2P) 정규화 값
    targets: dict[str, np.ndarray]    # split -> (n, M) int8
    summaries: dict[str, Any]


def build_dataset(config: dict[str, Any], seed: int) -> dict[str, list[SceneSpec]]:
    """시드별 데이터 생성 (독립 난수 스트림)."""
    dcfg_raw = dict(config["dataset"])
    novel = dcfg_raw.pop("novel_test")
    n_train = dcfg_raw.pop("n_train")
    n_dev = dcfg_raw.pop("n_dev")
    n_test = dcfg_raw.pop("n_test")
    dcfg = DatasetConfig(
        height=config["image"]["height"], width=config["image"]["width"],
        orientations_deg=tuple(config["circuit"]["orientations_deg"]), **dcfg_raw
    )
    rng = stream(seed, "data")
    scenes = {
        "train": generate_scenes(n_train, rng, dcfg, "train"),
        "dev": generate_scenes(n_dev, rng, dcfg, "dev"),
        "test": generate_scenes(n_test, rng, dcfg, "test"),
    }
    rng_novel = stream(seed, "novel_data")
    scenes["test_novel"] = generate_scenes(
        novel["n"], rng_novel, dcfg, "testnovel",
        contrast_range=tuple(novel["contrast_range"]),
        center_radius_range_frac=tuple(novel["center_radius_range_frac"]),
    )
    return scenes


def prepare_data(circuit: V1Circuit, scenes: dict[str, list[SceneSpec]],
                 teacher: LocalTeacher) -> PreparedData:
    """전처리 표본 계산 -> 훈련 데이터로만 정규화 계수 추정 -> 목표 생성.

    부작용: ``circuit.normalizer`` 를 훈련 표본으로 fit 한다.
    검증·시험 영상으로 계수를 다시 추정하지 않는다.
    """
    raw: dict[str, np.ndarray] = {}
    for split, ss in scenes.items():
        raw[split] = np.stack([circuit.preprocess(render_scene(s)) for s in ss], axis=0)
    circuit.normalizer.fit(raw["train"])
    drive = {k: circuit.normalizer.transform(v) for k, v in raw.items()}
    targets = {k: teacher.make_targets_batch(ss) for k, ss in scenes.items()}
    summaries = {k: split_summary(ss) for k, ss in scenes.items()}
    return PreparedData(scenes, raw, drive, targets, summaries)


class V1Experiment:
    """한 (조건, 시드) 실험."""

    def __init__(
        self,
        config: dict[str, Any],
        seed: int,
        condition: str,
        epochs: int,
        learning_rate: float | None = None,
        shared: dict[str, Any] | None = None,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"알 수 없는 조건: {condition}")
        self.config = config
        self.seed = int(seed)
        self.condition = condition
        self.epochs = int(epochs)
        self.lr = float(
            learning_rate if learning_rate is not None else config["learning"]["learning_rate"]
        )
        self.shared = shared or {}

    # ------------------------------------------------------------------
    def setup(self) -> None:
        cfg = self.config
        self.circuit = V1Circuit.build(cfg, self.seed, stream(self.seed, "init"))
        self.pop = self.circuit.population
        self.table = self.circuit.table
        self.engine = self.circuit.engine
        self.l2_ids = self.circuit.l2_ids
        self.l2_obs_tick = int(cfg["circuit"]["observe_ticks"]["L2"])
        self.l3_obs_tick = int(cfg["circuit"]["observe_ticks"]["L3"])
        self.l2_thresholds = self.pop.store.states[self.l2_ids, 1, 1].copy()
        self.desc = L2Descriptor.from_circuit(self.circuit)
        tcfg = cfg["teacher"]
        self.teacher = LocalTeacher(
            self.desc,
            orientation_tolerance_deg=tcfg["orientation_tolerance_deg"],
            distance_divisor=2.0,
            orientation_period_deg=tcfg["orientation_period_deg"],
        )
        self.relay = L1Relay(self.l2_ids)
        self.learner = LocalLearner(
            self.table,
            LearningConfig(
                epsilon=cfg["learning"]["epsilon"],
                learning_rate=self.lr,
                weight_max=cfg["learning"]["weight_max"],
            ),
        )
        scenes = self.shared.get("scenes") or build_dataset(cfg, self.seed)
        self.data = prepare_data(self.circuit, scenes, self.teacher)
        self.initial_weights = self.table.snapshot_weights()

    # ------------------------------------------------------------------
    def _free_trial(self, drive_values: np.ndarray, trial_id: Any):
        return self.engine.run_trial(
            (self.circuit.l4_drive_ids, drive_values), trial_id=trial_id
        )

    def predict_split(self, split: str) -> dict[str, np.ndarray]:
        """교사 없이 자유 실행만 해서 L2/L3 예측을 만든다 (가중치 불변)."""
        dv = self.data.drive[split]
        n = dv.shape[0]
        pred_l2 = np.zeros((n, self.l2_ids.size), dtype=np.int8)
        pred_l3 = np.zeros((n, self.circuit.l3_ids.size), dtype=np.int8)
        e_mean = np.zeros(n)
        i_mean = np.zeros(n)
        for i in range(n):
            r = self._free_trial(dv[i], trial_id=f"{split}/{i}")
            pred_l2[i] = r.observed_q[self.l2_ids]
            pred_l3[i] = r.observed_q[self.circuit.l3_ids]
            e_mean[i] = r.observed_E[self.l2_ids].mean()
            i_mean[i] = r.observed_I[self.l2_ids].mean()
        return {"l2": pred_l2, "l3": pred_l3, "E_mean": e_mean, "I_mean": i_mean}

    def predict_split_guided(self, split: str) -> np.ndarray:
        """교정을 켠 상태의 L2 활동 (참고용. **최종 성능이 아니다**)."""
        dv = self.data.drive[split]
        tg = self.data.targets[split]
        n = dv.shape[0]
        out = np.zeros((n, self.l2_ids.size), dtype=np.int8)
        margin = self.config["teacher"]["margin"]
        ovr = self.config["teacher"]["error_override_zeros"]
        for i in range(n):
            free = self._free_trial(dv[i], trial_id=f"{split}/guided/{i}")
            c = self.teacher.make_corrections(
                free.observed_u[self.l2_ids], tg[i], self.l2_thresholds, margin, ovr
            )
            routed = self.relay.route((self.l2_ids, c))
            g = self.engine.run_trial(
                (self.circuit.l4_drive_ids, dv[i]), correction=routed,
                trial_id=f"{split}/guided2/{i}"
            )
            out[i] = g.observed_q[self.l2_ids]
        return out

    # ------------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        cfg = self.config
        margin = cfg["teacher"]["margin"]
        ovr = cfg["teacher"]["error_override_zeros"]
        dv = self.data.drive["train"]
        tg_eval = self.data.targets["train"]
        n = dv.shape[0]

        # 훈련용 목표: shuffled_teacher 만 고정 무작위 순열로 표본 대응을 깨뜨린다.
        if self.condition == "shuffled_teacher":
            perm = stream(self.seed, "teacher_permutation").permutation(n)
            tg_train = tg_eval[perm]
        else:
            perm = None
            tg_train = tg_eval

        order_rng = stream(self.seed, "order")
        history: list[dict[str, Any]] = []
        n_trials = 0
        clip_num = 0
        clip_den = 0
        guided_correct = 0
        guided_total = 0
        free_correct = 0
        free_total = 0

        do_learn = self.condition in ("local_learning", "shuffled_teacher")
        do_correct = self.condition == "correction_only"

        if self.condition == "frozen":
            return {
                "n_training_trials": 0,
                "note_ko": "frozen 조건은 가중치를 바꾸지 않으므로 훈련 반복을 실행하지 않는다.",
                "epochs_run": 0,
                "history": [],
                "sample_order_stream": "order",
            }

        for ep in range(self.epochs):
            order = order_rng.permutation(n)
            ep_updates = 0
            ep_d_nonzero = 0
            for idx in order.tolist():
                free = self._free_trial(dv[idx], trial_id=f"train/e{ep}/{idx}")
                q_free = free.observed_q[self.l2_ids]
                u_free = free.observed_u[self.l2_ids]
                n_trials += 1
                free_correct += int(np.count_nonzero(q_free == tg_eval[idx]))
                free_total += q_free.size

                d = self.teacher.make_errors(tg_train[idx], q_free, ovr)
                ep_d_nonzero += int(np.count_nonzero(d))

                if do_correct:
                    c = self.teacher.make_corrections(
                        u_free, tg_train[idx], self.l2_thresholds, margin, ovr
                    )
                    routed = self.relay.route((self.l2_ids, c))
                    g = self.engine.run_trial(
                        (self.circuit.l4_drive_ids, dv[idx]), correction=routed,
                        trial_id=f"train/e{ep}/{idx}/guided"
                    )
                    qg = g.observed_q[self.l2_ids]
                    guided_correct += int(np.count_nonzero(qg == tg_train[idx]))
                    guided_total += qg.size
                    continue  # 가중치는 고정

                if do_learn:
                    snap = FreeRunSnapshot.from_trial(free, self.l2_obs_tick)
                    upd = self.learner.compute_updates(snap, (self.l2_ids, d))
                    st = self.learner.apply_updates(upd)
                    clip_num += st["n_clipped"]
                    clip_den += st["n_nonzero_G"]
                    ep_updates += 1 if st["changed"] else 0

            rec = {
                "epoch": ep,
                "n_weight_updates_applied": ep_updates,
                "mean_nonzero_errors_per_trial": ep_d_nonzero / max(1, n),
                "weight_version": self.table.weight_version,
                "weight_l2_norm": float(np.linalg.norm(self.table.weight)),
            }
            if (ep + 1) % max(1, self.epochs // 5) == 0 or ep == self.epochs - 1:
                pr = self.predict_split("dev")
                rec["dev_balanced_accuracy"] = evaluate_map(
                    self.data.targets["dev"], pr["l2"]
                )["balanced_accuracy"]
            history.append(rec)

        return {
            "n_training_trials": n_trials,
            "epochs_run": self.epochs,
            "history": history,
            "clipping_fraction_overall": (clip_num / clip_den) if clip_den else (
                None if not do_learn else 0.0),
            "n_clipped_total": clip_num,
            "n_nonzero_G_total": clip_den,
            "train_free_bit_accuracy_during_training": (
                free_correct / free_total if free_total else None
            ),
            "train_guided_bit_accuracy_during_training": (
                guided_correct / guided_total if guided_total else None
            ),
            "teacher_permutation_used": perm.tolist() if perm is not None else None,
            "sample_order_stream": "order",
        }

    # ------------------------------------------------------------------
    def evaluate(self, splits: Sequence[str] = ("dev", "test", "test_novel")) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for split in splits:
            pr = self.predict_split(split)
            y = self.data.targets[split]
            is_blank = [s.shape_type == "blank" for s in self.data.scenes[split]]
            oris = [s.orientation_deg for s in self.data.scenes[split]]
            pn = per_neuron_metrics(y, pr["l2"])
            out[split] = {
                "l2_overall": evaluate_map(y, pr["l2"]),
                "l3_overall": evaluate_map(y, pr["l3"]),
                "baselines": baseline_metrics(y),
                "blank": blank_false_firing(pr["l2"], is_blank),
                "breakdown": orientation_position_breakdown(
                    y, pr["l2"], self.circuit.l2_orientation,
                    self.circuit.grid.radial_bin[self.circuit.l2_point],
                ),
                "per_neuron": {
                    k: v for k, v in pn.items()
                    if k not in ("balanced_accuracy", "n_positive", "n_negative", "activity_rate")
                },
                "per_neuron_arrays": {
                    "balanced_accuracy": np.asarray(pn["balanced_accuracy"]).tolist(),
                    "n_positive": np.asarray(pn["n_positive"]).tolist(),
                    "activity_rate": np.asarray(pn["activity_rate"]).tolist(),
                },
                "E_mean_l2": float(pr["E_mean"].mean()),
                "I_mean_l2": float(pr["I_mean"].mean()),
                "scene_orientations": [None if o is None else float(o) for o in oris],
            }
            out[split]["_predictions"] = pr["l2"]
        return out

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        t0 = time.time()
        self.setup()
        t_setup = time.time() - t0

        # 학습 전 상태 진단 (초기값 적절성)
        pre = self.predict_split("train")
        pre_rate = float(pre["l2"].mean())

        t1 = time.time()
        train_info = self.train()
        t_train = time.time() - t1

        t2 = time.time()
        ev = self.evaluate()
        t_eval = time.time() - t2

        # 교정 중 성능 (참고용, 최종 성능 아님)
        guided = None
        if self.condition in ("correction_only", "local_learning", "shuffled_teacher"):
            gq = self.predict_split_guided("dev")
            guided = evaluate_map(self.data.targets["dev"], gq)

        post_rate = float(ev["dev"]["_predictions"].mean())
        w = self.table.weight
        tr = self.table.trainable
        result = {
            "condition": self.condition,
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.lr,
            "build_stats": self.circuit.build_stats,
            "normalizer": self.circuit.normalizer.to_dict(),
            "dataset_summaries": self.data.summaries,
            "initial_activity_rate_train_l2": pre_rate,
            "initial_weight_stats": {
                "mean": float(self.initial_weights[tr].mean()),
                "max": float(self.initial_weights[tr].max()),
                "min": float(self.initial_weights[tr].min()),
            },
            "final_weight_stats": {
                "mean": float(w[tr].mean()),
                "max": float(w[tr].max()),
                "min": float(w[tr].min()),
                "l2_norm": float(np.linalg.norm(w[tr])),
                "n_at_zero": int(np.count_nonzero(w[tr] <= 0.0)),
                "n_at_max": int(np.count_nonzero(w[tr] >= self.learner.cfg.weight_max)),
                "weight_version": self.table.weight_version,
                "bitwise_unchanged_from_init": bool(
                    np.array_equal(w.view(np.uint8), self.initial_weights.view(np.uint8))
                ),
            },
            "training": train_info,
            "evaluation": {k: {kk: vv for kk, vv in v.items() if kk != "_predictions"}
                           for k, v in ev.items()},
            "guided_dev_reference_only": guided,
            "timing_sec": {
                "setup": t_setup, "train": t_train, "eval": t_eval,
                "total": time.time() - t0,
            },
        }
        self._eval_raw = ev
        return result
