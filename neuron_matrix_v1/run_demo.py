#!/usr/bin/env python3
"""run_demo.py -- 저장한 모델을 불러와 **추론만** 수행하는 데모 (명세 [15]).

교사 목표는 데모의 자유 추론에 들어가지 않는다. ``--show-targets`` 는
분석 모드로 목표/불일치 지도를 **표시만** 한다.

사용법::

    python run_demo.py --model results/quick/models/local_learning_seed42
    python run_demo.py --model results/quick/models/local_learning_seed42 --headless
    python run_demo.py --model <경로> --headless --out results/quick/figures/demo

GUI 가 없는 환경에서는 ``--headless`` 로 고정 장면 PNG 를 저장한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets import SceneSpec, render_scene
from src.diagnostics import TraceRecorder
from src.persistence import load_model
from src.teacher import L2Descriptor, LocalTeacher
from src import visualization as viz

HERE = os.path.dirname(os.path.abspath(__file__))


def make_scene(orientation_deg: float, center_frac_x: float, center_frac_y: float,
               contrast: float, H: int, W: int, polarity: int = 1,
               length_frac: float = 0.7, line_width: float = 1.5,
               scene_id: str = "demo") -> SceneSpec:
    """데모용 장면 1개 (선 방향·위치·대비·극성을 직접 지정)."""
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    R = min(W - 1, H - 1) / 2.0
    mx = cx + center_frac_x * R
    my = cy + center_frac_y * R
    length = 2.0 * R * length_frac
    if abs(orientation_deg) < 45.0:
        dx, dy, ori, kind = length / 2.0, 0.0, 0.0, "horizontal_line"
    else:
        dx, dy, ori, kind = 0.0, length / 2.0, 90.0, "vertical_line"
    return SceneSpec(scene_id, scene_id, kind,
                     ((mx - dx, my - dy), (mx + dx, my + dy)),
                     ori, float(contrast), int(polarity), float(line_width), H, W)


def infer(circuit, scene: SceneSpec):
    """교사 없이 이미지 -> L2/L3 활동. 가중치를 바꾸지 않는다."""
    img = render_scene(scene)
    drive = circuit.drive_from_image(img)
    r = circuit.engine.run_trial(drive, trial_id=f"demo/{scene.scene_id}")
    return img, r


def headless(circuit, args, cfg) -> list[str]:
    H, W = cfg["image"]["height"], cfg["image"]["width"]
    desc = L2Descriptor.from_circuit(circuit)
    teacher = LocalTeacher(desc, cfg["teacher"]["orientation_tolerance_deg"])
    outdir = args.out or os.path.join(HERE, "results", "demo")
    os.makedirs(outdir, exist_ok=True)
    scenarios = [
        ("h_center_mid", 0.0, 0.0, 0.0, 0.35),
        ("v_center_mid", 90.0, 0.0, 0.0, 0.35),
        ("h_offset_hi", 0.0, 0.25, -0.45, 0.45),
        ("v_offset_lo", 90.0, -0.35, 0.20, 0.25),
    ]
    paths = []
    summary = []
    for name, ori, fx, fy, contrast in scenarios:
        sc = make_scene(ori, fx, fy, contrast, H, W, scene_id=f"demo-{name}")
        img, r = infer(circuit, sc)
        act = r.observed_q[circuit.l2_ids]
        tg = teacher.make_targets(sc) if args.show_targets else None
        p = viz.plot_activity_maps(
            circuit, act, os.path.join(outdir, f"demo_{name}.png"),
            targets=tg, analysis_mode=bool(args.show_targets),
            title=(f"저장 모델 추론 (학습 후) · 자유 실행 (교정 없음) · "
                   f"image_id={sc.scene_id} · seed={args.seed_label} · 관찰 tick=2 · "
                   f"ori={int(ori)}° contrast={contrast}"))
        paths.append(p)
        p2 = viz.plot_input_and_sampling(
            circuit, img, os.path.join(outdir, f"demo_{name}_input.png"),
            title_extra=f"image_id={sc.scene_id} · seed={args.seed_label}")
        paths.append(p2)
        summary.append({
            "scenario": name, "orientation_deg": ori, "contrast": contrast,
            "center_frac": [fx, fy],
            "n_l2_fired": int(act.sum()),
            "n_l2_fired_ori0": int(act[circuit.l2_orientation == 0.0].sum()),
            "n_l2_fired_ori90": int(act[circuit.l2_orientation == 90.0].sum()),
            "n_l3_fired": int(r.observed_q[circuit.l3_ids].sum()),
            "n_l4_fired": int(r.observed_q[circuit.l4_drive_ids].sum()),
            "n_target_positive": int(tg.sum()) if tg is not None else None,
        })

    # 선택 뉴런 진단 그림 (가장 입력이 많이 도착한 L2)
    sc = make_scene(0.0, 0.0, 0.0, 0.35, H, W, scene_id="demo-diag")
    img = render_scene(sc)
    drive = circuit.drive_from_image(img)
    probe = circuit.engine.run_trial(drive, trial_id="demo/probe")
    s = probe.arrivals_at(2)
    n_in = np.bincount(circuit.table.post_id, weights=(s > 0).astype(float),
                       minlength=circuit.table.n_neurons)[circuit.l2_ids]
    best = int(circuit.l2_ids[int(np.argmax(n_in))])
    rec = TraceRecorder(circuit.population, circuit.table, select_neurons=[best],
                        select_trials=["demo/diag"], max_rows=2000)
    circuit.engine.recorder = rec
    circuit.engine.run_trial(drive, trial_id="demo/diag")
    circuit.engine.recorder = None
    info = rec.inspect_neuron("demo/diag", best, 2)
    if info.get("found"):
        paths.append(viz.plot_neuron_state_and_contributions(
            circuit, info, os.path.join(outdir, "demo_neuron_state.png"),
            title=(f"저장 모델 추론 (학습 후) · 자유 실행 · image_id=demo-diag · "
                   f"seed={args.seed_label} · 관찰 tick=2")))
        summary.append({"inspected_neuron": circuit.population.neurons[best].name,
                        "observed": info["observed"] | {"top_contributions": len(info["observed"]["top_contributions"])},
                        "hypotheses": info["hypotheses"],
                        "reconstruction_check": info["reconstruction_check"]})

    with open(os.path.join(outdir, "demo_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "scenarios": summary,
                   "figures": paths,
                   "note_ko": "교사 목표는 표시용이며 학생의 자유 추론에 들어가지 않는다."},
                  f, ensure_ascii=False, indent=1, default=str)
    print(f"저장한 그림 {len(paths)}개 -> {outdir}")
    for s_ in summary[:4]:
        print(f"  {s_.get('scenario')}: L2 발화 {s_.get('n_l2_fired')} "
              f"(0°={s_.get('n_l2_fired_ori0')}, 90°={s_.get('n_l2_fired_ori90')})")
    return paths


def interactive(circuit, args, cfg) -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    H, W = cfg["image"]["height"], cfg["image"]["width"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    plt.subplots_adjust(bottom=0.32)
    sl_ori = Slider(plt.axes([0.15, 0.22, 0.7, 0.03]), "orientation(deg)", 0, 90, valinit=0, valstep=90)
    sl_x = Slider(plt.axes([0.15, 0.17, 0.7, 0.03]), "center x (R 배수)", -0.8, 0.8, valinit=0.0)
    sl_y = Slider(plt.axes([0.15, 0.12, 0.7, 0.03]), "center y (R 배수)", -0.8, 0.8, valinit=0.0)
    sl_c = Slider(plt.axes([0.15, 0.07, 0.7, 0.03]), "contrast", 0.05, 0.5, valinit=0.35)

    def redraw(_=None):
        sc = make_scene(sl_ori.val, sl_x.val, sl_y.val, sl_c.val, H, W, scene_id="demo-live")
        img, r = infer(circuit, sc)
        act = r.observed_q[circuit.l2_ids]
        g = circuit.grid
        for ax in axes:
            ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("입력 (추론만, 교사 없음)", fontsize=9)
        for o, ax in zip((0.0, 90.0), axes[1:]):
            m = circuit.l2_orientation == o
            pts = circuit.l2_point[m]
            ax.scatter(g.x[pts], g.y[pts], c=act[m], cmap="viridis", vmin=0, vmax=1, s=34)
            ax.invert_yaxis(); ax.set_aspect("equal")
            ax.set_title(f"L2 활동 (선호 {int(o)}°) 발화수={int(act[m].sum())}", fontsize=9)
        fig.suptitle(f"저장 모델 추론 · 자유 실행 · seed={args.seed_label} · 관찰 tick=2", fontsize=10)
        fig.canvas.draw_idle()

    for s in (sl_ori, sl_x, sl_y, sl_c):
        s.on_changed(redraw)
    redraw()
    plt.show()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="확장자 없는 체크포인트 접두사 (예 results/quick/models/local_learning_seed42)")
    ap.add_argument("--headless", action="store_true", help="GUI 없이 고정 장면 PNG 저장")
    ap.add_argument("--show-targets", action="store_true",
                    help="분석 모드: 교사 목표와 불일치 지도를 함께 표시 (추론에는 미사용)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    circuit = load_model(args.model)
    cfg = circuit.config
    with open(f"{args.model}.json", encoding="utf-8") as f:
        meta = json.load(f)
    args.seed_label = meta.get("extra", {}).get("seed", "?")
    print(f"모델 로드: {args.model}  ({time.time()-t0:.2f}s)")
    print(f"  뉴런 {circuit.build_stats['n_neurons']}개, 연결 {circuit.build_stats['n_connections']}개, "
          f"weight_version={circuit.table.weight_version}")
    print(f"  조건={meta.get('extra', {}).get('condition')} seed={args.seed_label} "
          f"epochs={meta.get('extra', {}).get('epochs')}")

    if args.headless:
        headless(circuit, args, cfg)
        return 0
    try:
        interactive(circuit, args, cfg)
    except Exception as exc:  # GUI 없음 등
        print(f"[알림] 대화형 GUI 를 열 수 없어 headless 로 전환한다: {exc}")
        headless(circuit, args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
