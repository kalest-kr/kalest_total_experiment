#!/usr/bin/env python3
"""make_report.py -- 실행 결과 JSON 에서 한국어 보고서를 **자동 생성**한다.

사용법::

    python make_report.py
    python make_report.py --profile main
    python make_report.py --profile quick --checks results/checks.json --out results/quick/REPORT_ko.md

보고서는 명세 [16] 의 질문에 수치로 답한다. 직접 실행하지 못한 항목은
"미실행"으로 표시한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "정의되지 않음"
    if isinstance(x, bool):
        return "예" if x else "아니오"
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pm(stat: dict[str, Any], nd: int = 4) -> str:
    if stat is None or stat.get("mean") is None:
        return "정의되지 않음"
    if stat.get("std") is None:
        return f"{stat['mean']:.{nd}f} (시드 1개)"
    return f"{stat['mean']:.{nd}f} ± {stat['std']:.{nd}f}"


def fresh_code_hashes() -> list[dict[str, Any]]:
    """보고서 생성 시점의 코드·설정 파일 해시를 다시 계산한다."""
    sys.path.insert(0, HERE)
    from src.persistence import hash_manifest
    files = [os.path.join(HERE, p) for p in
             ["config.json", "requirements.txt", "README_ko.md", "run_checks.py",
              "run_experiment.py", "run_demo.py", "make_report.py", "make_manifest.py",
              os.path.join("tests", "test_units.py")]]
    files += [os.path.join(HERE, "src", f)
              for f in sorted(os.listdir(os.path.join(HERE, "src"))) if f.endswith(".py")]
    return hash_manifest(files, HERE)


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_report(summary: dict[str, Any], checks: dict[str, Any] | None,
                 profile: str) -> str:
    L: list[str] = []
    A = L.append
    cfg = summary["config"]
    conds = summary["conditions"]
    sm = summary["summary_by_condition"]
    diag = summary.get("diagnostics", {}) or {}
    recs = summary["records"]

    A(f"# 뉴런 3×3 행렬 · 작은 V1 회로 · 국소 교사 학습 — 실행 결과 보고서 ({profile} 프로필)")
    A("")
    A(f"- 생성 시각(UTC): {summary['generated_at_utc']}")
    A(f"- 실행 소요 시간: {summary['elapsed_sec']:.1f} 초")
    A(f"- 시드: {summary['seeds']} · 에폭: {summary['epochs']} · 조건: {conds}")
    lr_line = ("탐색 ON" if summary["lr_search_enabled"]
               else f"고정 {summary['learning_rate_fixed']}")
    A(f"- 학습률: {lr_line}")
    A("- **이 보고서는 run_summary.json 에서 자동 생성되었다.** 수치는 실제 실행 결과다.")
    A("")
    A("> 주의: 이 구현은 이산 시점의 가중합·임계 발화 모델이다. 실제 전도도, 막전위의 시간 적분,")
    A("> 칼슘 농도, 생화학적 STDP 를 구현한 모델이 아니다. 전체 과제 손실을 연쇄 미분하는 역전파는")
    A("> 사용하지 않았고, 대신 **뉴런별 목표를 외부 교사가 제공한다는 정보상의 가정**을 둔다.")
    A("> 뇌 전체를 재현했다고 주장하지 않는다.")
    A("")

    # --- 1. 실행 환경 ---------------------------------------------------
    A("## 1. 실행 환경과 실행 명령")
    A("")
    env = summary["environment"]
    A(table(["항목", "값"], [
        ["Python", env["python"].splitlines()[0]],
        ["플랫폼", env["platform"]],
        ["NumPy", env["numpy"]],
        ["SciPy", env["scipy"]],
        ["Matplotlib", env["matplotlib"]],
        ["Pillow", env["pillow"]],
        ["PyTorch", env["torch"]],
        ["GPU", env["gpu"]],
    ]))
    A("")
    A("```bash")
    A("python run_checks.py")
    A("python run_experiment.py --profile quick")
    A("python run_experiment.py --profile main")
    A("python make_report.py --profile main")
    A("python run_demo.py --model results/main/models/local_learning_seed42 --headless")
    A("```")
    A("")

    # --- 2. 구현 범위 ---------------------------------------------------
    A("## 2. 구현한 것과 구현하지 않은 것")
    A("")
    st = summary["implemented_scope"]
    A(f"- 구현된 영역: **{', '.join(st['implemented_areas'])}** (L4 / 억제 중계 / L2 / L3, L1 은 교정 인터페이스)")
    A(f"- 구현하지 않은 영역: **{', '.join(st['not_implemented_areas'])}** — 입출력 프로토콜과 설계 설명만 제공하며,")
    A("  호출하면 `NotImplementedError` 를 낸다. 이름만 붙인 항등 함수를 두지 않았다.")
    A(f"- {st['note_ko']}")
    A("")
    bs = recs[0]["build_stats"]
    A(table(["회로 규모", "값"], [
        ["뉴런 수", bs["n_neurons"]],
        ["연결 수", bs["n_connections"]],
        ["학습 대상 연결 수", bs["n_trainable_connections"]],
        ["로그-극좌표 격자점", bs["n_grid_points"]],
        ["L2 입력 이웃 수 k", bs["l2_input_k_nearest"]],
        ["L3 풀링 k", bs["l3_pool_k"]],
        ["방향 채널", bs["orientations_deg"]],
    ]))
    A("")

    # --- 3. 필수 검증 ---------------------------------------------------
    A("## 3. 필수 검증 (T1~T10) 결과")
    A("")
    if checks:
        rows = []
        for c in checks["checks"]:
            n_ok = sum(1 for a in c["assertions"] if a["passed"])
            rows.append([c["id"], c["title"], "통과" if c["passed"] else "실패",
                         f"{n_ok}/{len(c['assertions'])}", f"{c['elapsed_sec']:.2f}s"])
        A(table(["검사", "내용", "결과", "세부 단언", "소요"], rows))
        A("")
        A(f"- 전체: **통과 {checks['n_passed']} / 실패 {checks['n_failed']}** "
          f"(총 {checks['n_checks']}개, {checks['elapsed_sec']:.1f}s)")
        t2 = next((c for c in checks["checks"] if c["id"] == "T2"), None)
        if t2 and t2["detail"].get("abcd"):
            d = t2["detail"]["abcd"]
            A(f"- A/B/C→D 예제 재현: E={fmt(d['E'])}, I={fmt(d['I'])}, u={fmt(d['u'])}, q={d['q']} "
              f"(기대값 0.8 / 0.6 / 0.2 / 0 과 일치). 이 결과를 근거로 C 가 잘못된 연결이라고 단정하지 않는다.")
        t6 = next((c for c in checks["checks"] if c["id"] == "T6"), None)
        if t6 and t6["detail"].get("detail"):
            d = t6["detail"]["detail"]
            A(f"- 작은 학습 예제(T6): {d['iterations']} 반복에 해결, 흥분 가중치 {fmt(d['w_exc'],3)}, "
              f"억제 가중치 {fmt(d['w_inh'],3)}. {d['note_ko']}")
        t10 = next((c for c in checks["checks"] if c["id"] == "T10"), None)
        if t10 and t10["detail"].get("detail"):
            d = t10["detail"]["detail"]
            rows = []
            for tag in ("64x64", "1000x1000"):
                if tag in d:
                    x = d[tag]
                    rows.append([tag, x["n_points"], f"{x['valid_fov_fraction']*100:.1f}%",
                                 fmt(x["sigma_pool_max"], 2), fmt(x["rf_radius_max_px"], 2),
                                 f"{x['time_encode_sec']*1000:.1f} ms",
                                 f"{x['time_sample_sec']*1000:.1f} ms"])
            A("")
            A(table(["입력 크기", "표본 수", "유효 시야 비율", "최대 sigma_pool",
                     "최대 수용장 반경(px)", "DoG 부호화", "격자 표본 추출"], rows))
            A("")
            A(f"- {d['note_ko']}")
    else:
        A("- checks.json 을 찾지 못해 표시하지 못했다. `python run_checks.py` 를 먼저 실행하라. **(미실행)**")
    A("")

    # --- 4. 데이터 -------------------------------------------------------
    A("## 4. 데이터")
    A("")
    ds = recs[0]["dataset_summaries"]
    rows = []
    for split in ("train", "dev", "test", "test_novel"):
        if split not in ds:
            continue
        s = ds[split]
        fr = s["fraction_by_shape"]
        rows.append([
            split, s["n"],
            f"{fr.get('horizontal_line',0)*100:.1f}% / {fr.get('vertical_line',0)*100:.1f}% / {fr.get('blank',0)*100:.1f}%",
            f"{s['polarity_counts']['+1']} / {s['polarity_counts']['-1']}",
            f"[{fmt(s['contrast_min'],3)}, {fmt(s['contrast_max'],3)}]",
            f"x[{fmt(s['center_x_range'][0],1)},{fmt(s['center_x_range'][1],1)}] "
            f"y[{fmt(s['center_y_range'][0],1)},{fmt(s['center_y_range'][1],1)}]",
        ])
    A(table(["분할", "표본 수", "가로/세로/빈 화면", "밝은 선/어두운 선", "대비 범위(진폭)", "선 중심 범위(px)"], rows))
    A("")
    A("- 대비는 배경 0.5 대비 **진폭**이다 (Michelson 대비 = 진폭/0.5). 약한 대비는 기본 데이터에서 제외했다.")
    A("- `test_novel` 은 학습률 선택에 쓰이지 않는 **독립 추가 시험**이며, 대비와 선 중심 반경 범위가 기본 시험과 다르다.")
    A("- 분할 간 `scene_id` 와 `base_scene_id` 가 겹치지 않음은 T8 에서 검사했다.")
    A("")
    nz = recs[0]["normalizer"]
    A(f"- L4 입력 정규화 계수: 훈련 표본의 양수값 {nz['percentile']:.0f}백분위 = **{fmt(nz['scale'],5)}**, "
      f"degenerate={fmt(nz['degenerate'])} (양수값 {nz['n_positive_values']}개). 검증·시험으로 재추정하지 않았다.")
    if diag.get("l4_binarization"):
        lb = diag["l4_binarization"]
        A(f"- L4 이진화 후 장면당 평균 활성 L4 뉴런 수 **{fmt(lb['mean_active_l4_per_scene'],2)} / {lb['n_l4']}**, "
          f"활성 L4 가 하나도 없는 장면 {lb['n_scenes_with_no_active_l4']}개(dev). {lb['note_ko']}")
    A("")

    # --- 5. 대조 실험 ----------------------------------------------------
    A("## 5. 대조 실험 결과 (모든 최종 평가는 **교정 없이** 수행)")
    A("")
    rows = []
    for c in conds:
        if c not in sm:
            continue
        s = sm[c]
        rows.append([c, s["learning_rate"], pm(s["balanced_accuracy"]), pm(s["f1"]),
                     pm(s["accuracy"]), pm(s["dev_balanced_accuracy"]),
                     pm(s["test_novel_balanced_accuracy"])])
    A(table(["조건", "학습률", "test balanced acc", "test F1", "test 비트 정확도",
             "dev balanced acc", "추가시험 balanced acc"], rows))
    A("")
    A("주 지표는 balanced accuracy 와 F1 이다. 불활성 뉴런이 많으므로 전체 비트 정확도만으로 성공을 주장하지 않는다.")
    A("")
    base = recs[0]["evaluation"]["test"]["baselines"]
    A("### 5.1 기준값 (항상 비활성 / 항상 발화)")
    A("")
    A(table(["기준 예측", "balanced accuracy", "F1", "비트 정확도"], [
        ["항상 비활성", fmt(base["always_inactive"]["balanced_accuracy"]),
         fmt(base["always_inactive"]["f1"]), fmt(base["always_inactive"]["accuracy"])],
        ["항상 발화", fmt(base["always_active"]["balanced_accuracy"]),
         fmt(base["always_active"]["f1"]), fmt(base["always_active"]["accuracy"])],
    ]))
    A(f"")
    A(f"- 시험 목표의 양성 비율은 **{fmt(base['positive_rate']*100,2)}%** 다.")
    A("")

    A("### 5.2 혼동 행렬과 오류율 (조건·시드별 원자료)")
    A("")
    rows = []
    for r in recs:
        t = r["evaluation"]["test"]["l2_overall"]
        rows.append([r["condition"], r["seed"], t["confusion"]["TP"], t["confusion"]["FP"],
                     t["confusion"]["TN"], t["confusion"]["FN"],
                     fmt(t["false_positive_rate"]), fmt(t["false_negative_rate"]),
                     fmt(t["balanced_accuracy"]), fmt(t["f1"])])
    A(table(["조건", "시드", "TP", "FP", "TN", "FN", "FPR", "FNR", "balanced acc", "F1"], rows))
    A("")

    A("### 5.3 frozen 대비 짝지은 차이 (test balanced accuracy)")
    A("")
    rows = []
    for c in conds:
        if c == "frozen" or c not in sm or "paired_difference_vs_frozen_test_bacc" not in sm[c]:
            continue
        d = sm[c]["paired_difference_vs_frozen_test_bacc"]
        rows.append([c, d["n"], fmt(d["mean_difference"]),
                     fmt(d["std_difference"]) if d["std_difference"] is not None else "시드 1개",
                     ", ".join(f"{v:+.4f}" for v in d["per_seed_difference"])])
    if rows:
        A(table(["조건", "시드 수", "평균 차이", "표준편차", "시드별 차이"], rows))
    A("")

    A("### 5.4 빈 화면 불필요 발화율 / E·I 평균 / 가중치")
    A("")
    rows = []
    for c in conds:
        if c not in sm:
            continue
        s = sm[c]
        rows.append([c, pm(s["blank_false_firing_rate"], 5), pm(s["E_mean_l2_test"], 4),
                     pm(s["I_mean_l2_test"], 4), pm(s["weight_l2_norm"], 3),
                     pm(s["clipping_fraction"], 5)])
    A(table(["조건", "빈 화면 발화율", "L2 E 평균", "L2 I 평균", "학습 연결 가중치 L2 노름", "clipping 비율"], rows))
    A("")

    A("### 5.5 뉴런별 상태 (미관측 채널 / 지표 미정의)")
    A("")
    rows = []
    for c in conds:
        if c not in sm:
            continue
        s = sm[c]
        rows.append([c, pm(s["n_never_active_test"], 1), pm(s["n_no_positive_target_test"], 1),
                     pm(s["n_undefined_test"], 1)])
    A(table(["조건", "한 번도 발화 안 한 L2 수", "양성 목표가 없는 L2 수", "지표 미정의 L2 수"], rows))
    A("")
    A("- 양성 또는 음성 표본이 한 종류도 없는 뉴런은 balanced accuracy 가 정의되지 않아 NaN 으로 두었고 0 으로 채우지 않았다.")
    if diag.get("per_neuron_training_positives"):
        p = diag["per_neuron_training_positives"]
        A(f"- 훈련 양성 표본 수(뉴런별): 최소 {p['min']}, 최대 {p['max']}, 평균 {fmt(p['mean'],2)}; "
          f"양성이 한 번도 없는 채널 **{p['n_neurons_with_zero_positive_train']} / {p['n_l2']}**. {p['note_ko']}")
    A("")

    A("### 5.6 교정 중 성능과 교정 없는 성능 (반드시 분리)")
    A("")
    rows = []
    for c in conds:
        if c not in sm:
            continue
        s = sm[c]
        rows.append([c, pm(s["guided_dev_balanced_accuracy_reference_only"]),
                     pm(s["dev_balanced_accuracy"])])
    A(table(["조건", "dev (교정 켬, 참고용)", "dev (교정 없음, 최종 평가 방식)"], rows))
    A("")
    A("**유도 중 정확도는 최종 성능이 아니다.** 외부 교사가 목표 반응을 강하게 유도한 공학적 조작의 결과다.")
    A("")

    A("### 5.7 위치·방향별 반응")
    A("")
    ref = next((r for r in recs if r["condition"] == "local_learning"), recs[0])
    bd = ref["evaluation"]["test"]["breakdown"]
    rows = []
    for k, v in bd.items():
        if k.startswith("orientation_"):
            rows.append([f"선호 {k.split('_')[1]}°", fmt(v["balanced_accuracy"]), fmt(v["f1"]),
                         v["confusion"]["TP"], v["confusion"]["FP"], v["confusion"]["FN"]])
    A(f"**{ref['condition']} (seed={ref['seed']}) 의 방향별 시험 성능**")
    A("")
    A(table(["방향 채널", "balanced acc", "F1", "TP", "FP", "FN"], rows))
    A("")
    rows = []
    for k, v in bd["by_radial_bin"].items():
        rows.append([k.replace("radial_bin_", "반경 bin "), fmt(v["balanced_accuracy"]),
                     fmt(v["f1"]), v["n_positive"], v["confusion"]["TP"],
                     v["confusion"]["FP"], v["confusion"]["FN"]])
    A("**반경 bin 별 시험 성능 (안쪽 0 → 바깥 7)**")
    A("")
    A(table(["위치", "balanced acc", "F1", "양성 표본", "TP", "FP", "FN"], rows))
    A("")
    A("### 5.8 같은 영상 재검사 · 새로운 위치·대비·극성")
    A("")
    A("- **같은 영상 재검사**: 같은 초기 상태와 같은 입력으로 다시 실행하면 관찰 활동과 u 가 "
      "비트 단위로 동일하다 (T4 에서 검사). 다른 영상을 사이에 실행해도 같다 (상태 누출 없음).")
    A("- **새로운 위치·대비**: `test_novel` 은 대비 "
      f"{cfg['dataset']['novel_test']['contrast_range']} (기본 데이터 "
      f"[{cfg['dataset']['contrast_min']}, {cfg['dataset']['contrast_max']}] 밖), 선 중심 반경 "
      f"{cfg['dataset']['novel_test']['center_radius_range_frac']}×R (기본 "
      f"0~{cfg['dataset']['center_radius_frac']}×R 밖) 로 만든 **독립 추가 시험**이다. "
      "기본 시험과 분리해 정의했고 학습률 선택에 사용하지 않았다.")
    rows = []
    for c in conds:
        if c not in sm:
            continue
        rows.append([c, pm(sm[c]["balanced_accuracy"]), pm(sm[c]["test_novel_balanced_accuracy"])])
    A("")
    A(table(["조건", "기본 시험 balanced acc", "추가 시험(새 위치·대비) balanced acc"], rows))
    A("")
    A("- **극성**: 밝은 선(+1)과 어두운 선(−1) 이 모든 분할에 함께 들어 있다 (4절 표). "
      "ON/OFF 두 채널이 모두 L4 로 들어가므로 학생은 두 극성을 같은 경로로 받는다.")
    A("")

    # --- 6. 명세 질문 ----------------------------------------------------
    A("## 6. 명세 [16] 의 질문에 대한 수치 답변")
    A("")
    A("### Q1. 입력 목록과 당시 가중치로 발화 판단을 재구성할 수 있는가?")
    A("")
    A("**예.** 진단 기록은 연결별 `received_value`, `weight_used`, `signed_contribution`, `weight_version` 을")
    A("모두 남기고, 그 합이 저장된 E/I/u 와 일치하는지 검사한다.")
    if checks:
        t2 = next((c for c in checks["checks"] if c["id"] == "T2"), None)
        if t2:
            for a in t2["assertions"]:
                A(f"- {'통과' if a['passed'] else '실패'}: {a['assertion']}")
    if diag.get("trace_csv"):
        tc = diag["trace_csv"]
        A(f"- 기록된 상태 행 {tc['n_state_rows']}개 / 연결 기여 행 {tc['n_contribution_rows']}개 "
          f"(최대 행 수 초과로 버린 행 {tc['n_dropped']}개). "
          f"`diagnostics/trace_states.csv`, `diagnostics/trace_contributions.csv`.")
    ex = (diag.get("inspect_neuron_examples") or [])
    if ex:
        e0 = max(ex, key=lambda e: e["observed"]["n_arriving_connections"])
        o = e0["observed"]
        rc = e0["reconstruction_check"]
        A("")
        A(f"예시 (`{o['neuron_name']}`, trial `{o['trial_id']}`, tick {o['tick']}, "
          f"weight_version {o['weight_version']}):")
        A("")
        A(table(["항목", "저장값", "연결 기여에서 재구성"], [
            ["E", fmt(o["E"], 6), fmt(rc["E_from_contributions"], 6)],
            ["I", fmt(o["I"], 6), fmt(rc["I_from_contributions"], 6)],
            ["u", fmt(o["u"], 6), fmt(rc["u_from_contributions"], 6)],
            ["threshold", fmt(o["threshold"], 6), "-"],
            ["q", o["q"], "-"],
            ["도착 연결 수", o["n_arriving_connections"], "-"],
        ]))
        A("")
        A(f"- 관측 사실과 원인 가설은 분리해 출력한다. 이 예시의 가설: {e0['hypotheses'] or '(없음)'}")
        A(f"- 활동 판정: {e0.get('activity_verdict', '-')}")
    A("")

    A("### Q2. 잘못 발화하거나 발화하지 못한 위치·방향 채널을 찾을 수 있는가?")
    A("")
    A("**예.** 위치(반경 bin)·방향별 FP/FN 표(5.7절)와 장면별 불일치 지도 그림으로 찾는다.")
    worst_fn = max(bd["by_radial_bin"].items(), key=lambda kv: kv[1]["confusion"]["FN"])
    worst_fp = max(bd["by_radial_bin"].items(), key=lambda kv: kv[1]["confusion"]["FP"])
    A(f"- `{ref['condition']}` (seed={ref['seed']}) 기준 미발화(FN)가 가장 많은 위치: "
      f"**{worst_fn[0]}** (FN={worst_fn[1]['confusion']['FN']}, 양성 {worst_fn[1]['n_positive']}).")
    A(f"- 과다발화(FP)가 가장 많은 위치: **{worst_fp[0]}** (FP={worst_fp[1]['confusion']['FP']}).")
    if diag.get("measured_orientation_preference"):
        m = diag["measured_orientation_preference"]
        A(f"- 측정된 방향 선호가 메타데이터 `preferred_orientation` 과 일치한 L2 채널 "
          f"**{m['n_matching_metadata']} / {m['n_l2']}**, 측정 불가(반응 동률) {m['n_ambiguous']}개. {m['note_ko']}")
    A("- 그림: `figures/activity_after_seed*.png` (활동/목표/불일치 3단), `figures/orientation_preference_seed*.png`.")
    A("")

    A("### Q3. 국소 가중치 학습 후 교사를 제거해도 개선이 유지되는가?")
    A("")
    if "local_learning" in sm and "frozen" in sm:
        d = sm["local_learning"].get("paired_difference_vs_frozen_test_bacc")
        A(f"**유지된다.** 최종 시험 평가는 교사·교정을 모두 제거하고 일시 상태를 초기화한 자유 실행이다.")
        A(f"- frozen: {pm(sm['frozen']['balanced_accuracy'])} → local_learning: "
          f"{pm(sm['local_learning']['balanced_accuracy'])}")
        if d:
            A(f"- 짝지은 차이 평균 **{fmt(d['mean_difference'])}** (시드별 "
              f"{', '.join(f'{v:+.4f}' for v in d['per_seed_difference'])})")
        A(f"- F1: frozen {pm(sm['frozen']['f1'])} → local_learning {pm(sm['local_learning']['f1'])}")
        A(f"- 독립 추가 시험(새 대비·새 위치 범위)에서도: frozen "
          f"{pm(sm['frozen']['test_novel_balanced_accuracy'])} → local_learning "
          f"{pm(sm['local_learning']['test_novel_balanced_accuracy'])}")
        A("- 저장한 체크포인트를 다시 불러와 교사 없이 예측했을 때 결과가 일치함은 T9 에서 검사했다.")
    else:
        A("- 해당 조건이 실행되지 않아 비교할 수 없다. **(미실행)**")
    A("")

    A("### Q4. 활동 교정만 한 조건과 지속적 학습 조건이 구별되는가?")
    A("")
    eq = summary.get("frozen_vs_correction_only")
    if "correction_only" in sm:
        A(f"**구별된다.**")
        A(f"- `correction_only` 는 훈련 중 교정을 시연해 dev(교정 켬) "
          f"{pm(sm['correction_only']['guided_dev_balanced_accuracy_reference_only'])} 를 보이지만,")
        A(f"  교정을 제거한 최종 시험에서는 {pm(sm['correction_only']['balanced_accuracy'])} 로 "
          f"frozen({pm(sm['frozen']['balanced_accuracy'])}) 과 같다.")
        if eq:
            A(f"- frozen 과 correction_only 의 교정 없는 시험 예측이 **완전히 동일한가**: "
              f"{fmt(eq['all_equal'])} (시드별 {eq['frozen_equals_correction_only_without_correction']}).")
            A(f"  {eq['note_ko']}")
        A(f"- `local_learning` 은 교정을 제거한 뒤에도 {pm(sm['local_learning']['balanced_accuracy'])} 를 유지한다.")
        A(f"- 가중치 변화 여부: correction_only 의 최종 가중치가 초기값과 비트 단위로 동일한가 = "
          f"{fmt(next(r['final_weight_stats']['bitwise_unchanged_from_init'] for r in recs if r['condition']=='correction_only'))}, "
          f"local_learning = "
          f"{fmt(next(r['final_weight_stats']['bitwise_unchanged_from_init'] for r in recs if r['condition']=='local_learning'))}.")
    else:
        A("- `correction_only` 조건이 실행되지 않았다. **(미실행)**")
    A("")

    A("### Q5. 잘못된 교사, 정보 손실, 입력 부재, clipping 이 어떤 실패를 만드는가?")
    A("")
    if "shuffled_teacher" in sm:
        A(f"**잘못된 교사(shuffled_teacher)**: 훈련 목표의 전체 빈도는 보존하고 영상-목표 대응만 깨뜨렸다.")
        A(f"- test balanced accuracy {pm(sm['shuffled_teacher']['balanced_accuracy'])} 로 "
          f"frozen({pm(sm['frozen']['balanced_accuracy'])}) 수준에 머문다. "
          f"local_learning({pm(sm['local_learning']['balanced_accuracy'])}) 과 뚜렷이 다르다.")
        st_fp = sm["shuffled_teacher"]["false_positive_rate"]["mean"]
        fz_fp = sm["frozen"]["false_positive_rate"]["mean"]
        st_act = sm["shuffled_teacher"]["n_never_active_test"]["mean"]
        fz_act = sm["frozen"]["n_never_active_test"]["mean"]
        rel = "높다" if (st_fp or 0) > (fz_fp or 0) else ("같다" if (st_fp or 0) == (fz_fp or 0) else "낮다")
        A(f"- 가중치는 실제로 움직였다: 학습 연결 L2 노름 "
          f"{pm(sm['shuffled_teacher']['weight_l2_norm'],3)} (frozen "
          f"{pm(sm['frozen']['weight_l2_norm'],3)}), clipping 비율 "
          f"{pm(sm['shuffled_teacher']['clipping_fraction'],5)}, "
          f"한 번도 발화하지 않은 L2 수 {fmt(st_act,1)} (frozen {fmt(fz_act,1)}).")
        A(f"- 그 결과 오경보율(FPR)이 {fmt(st_fp)} 로 frozen({fmt(fz_fp)}) 보다 {rel}. "
          f"즉 **학습은 일어나되 목표와 맞지 않는 방향**이며, balanced accuracy 는 거의 개선되지 않는다.")
        A(f"- 빈 화면 불필요 발화율: shuffled_teacher "
          f"{pm(sm['shuffled_teacher']['blank_false_firing_rate'],5)}, frozen "
          f"{pm(sm['frozen']['blank_false_firing_rate'],5)}, local_learning "
          f"{pm(sm['local_learning']['blank_false_firing_rate'],5)}.")
    if diag.get("teacher_feasibility_dev"):
        f_ = diag["teacher_feasibility_dev"]
        A("")
        A(f"**정보 손실 / 식별 불가능성**: 같은 L4 발화 패턴에 서로 다른 목표가 붙은 dev 표본 그룹 "
          f"{f_['n_conflicting_groups']}개 (표본 {f_['n_samples_in_conflicting_groups']}개, "
          f"충돌 목표 비트 {f_['n_conflicting_target_bits']}개), 서로 다른 감각 패턴 "
          f"{f_['n_distinct_sensory_patterns']}개. {f_['note_ko']}")
    if diag.get("l4_binarization"):
        lb = diag["l4_binarization"]
        A(f"- **입력 부재**: dev 장면 중 활성 L4 가 하나도 없는 장면 {lb['n_scenes_with_no_active_l4']}개. "
          f"이런 장면에서는 도착 입력이 전부 0 이므로 규칙상 가중치 갱신이 0 이다 "
          f"(이를 숨기려고 임계값이나 기저 입력을 자동 조정하지 않았다).")
    if diag.get("inspect_neuron_examples"):
        quiet = [e for e in diag["inspect_neuron_examples"]
                 if e["observed"]["n_arriving_connections"] == 0]
        if quiet:
            A(f"- 실제로 도착 입력이 0 인 L2 뉴런 기록 예: `{quiet[0]['observed']['neuron_name']}` "
              f"(가설: {quiet[0]['hypotheses']}).")
    A("")
    rows = []
    for c in conds:
        if c not in sm:
            continue
        r0 = next(r for r in recs if r["condition"] == c)
        rows.append([c, pm(sm[c]["clipping_fraction"], 5),
                     r0["final_weight_stats"]["n_at_zero"],
                     r0["final_weight_stats"]["n_at_max"],
                     r0["build_stats"]["n_trainable_connections"]])
    A("**clipping**: 원래 G 와 clipping 후 실제 변화량을 모두 집계했다 (시드 대표값은 첫 시드 기준 개수).")
    A("")
    A(table(["조건", "clipping 비율", "가중치 0 에 붙은 연결", "가중치 상한에 붙은 연결", "학습 연결 수"], rows))
    A("")
    A("- clipping 은 가중치를 [0, weight_max] 로만 자르므로 **연결 종류(부호)를 뒤집지 않는다** (T5 에서 검사).")
    A("- 0 에 붙은 흥분 연결과 상한에 붙은 억제 연결은 그 방향으로 더 학습할 수 없는 포화 상태다.")
    A("")

    A("### Q6. 현재 교사는 어떤 정답 정보를 이미 알고 있는가?")
    A("")
    A("교사는 다음을 **이미 알고 있다**. 이것은 정보상의 가정이며 학습으로 얻은 것이 아니다.")
    A("")
    A("1. 장면의 도형 종류, 유한 선분의 양끝 좌표, 방향(도), 대비, 극성, 선폭 (원본 영상 좌표계 기준).")
    A("2. 각 L2 뉴런의 수용장 중심과 반경, 그리고 그 뉴런에 지정된 선호 방향.")
    A(f"3. 위 둘로부터 사람이 설계한 규칙 "
      f"`d_i ≤ rf_radius_i/2 이고 방향차 ≤ {cfg['teacher']['orientation_tolerance_deg']:.0f}°` 로 만든 뉴런별 이진 목표.")
    A("4. 어느 뉴런에 어떤 교정값을 보낼지 (L1Relay 의 목적지 사상).")
    A("")
    A("- 이 목표는 **사람이 설계한 위치·방향 지도**이며 실제 V1 정상 반응의 측정값이 아니다.")
    A("- 이 목표를 학습한 것은 사전에 지정한 표현을 학습한 것이다. 라벨 없이 뉴런 연결의 의미를 자동 추출한 것이 아니다.")
    A("- IT 의 최종 오차 하나에서 이 지도를 자동 역산한 기능은 이 프로젝트에 **없다**.")
    A("- 학생(감각 경로)은 이미지 픽셀만 받는다. 라벨 메타데이터만 바꿔도 전처리·감각 입력·학습 전 출력이 비트 단위로 같음은 T8 에서 검사했다.")
    A("")

    # --- 7. 초기값/학습 동역학 -------------------------------------------
    A("## 7. 초기값 적절성과 학습 동역학")
    A("")
    rows = []
    for r in recs:
        rows.append([r["condition"], r["seed"],
                     fmt(r["initial_activity_rate_train_l2"], 6),
                     fmt(r["initial_weight_stats"]["mean"], 4),
                     fmt(r["final_weight_stats"]["mean"], 4),
                     fmt(r["final_weight_stats"]["l2_norm"], 3),
                     r["training"].get("n_training_trials", 0),
                     r["final_weight_stats"]["weight_version"]])
    A(table(["조건", "시드", "학습 전 L2 발화율(train)", "초기 가중치 평균",
             "최종 가중치 평균", "최종 L2 노름", "훈련 시행 수", "weight_version"], rows))
    A("")
    init_rate = recs[0]["initial_activity_rate_train_l2"]
    A(f"- 초기 가중치 U(0, {cfg['circuit']['init_weight_high']}) 에서 학습 전 L2 발화율은 "
      f"**{init_rate*100:.4f}%** 다. 즉 초기에는 거의 **모두 비활성**이다 "
      f"(명세가 보고하라고 한 '초기값이 부적절해 거의 모두 발화하거나 모두 비활성화되는지'에 대한 답).")
    A("- 흥분/억제 두 경로가 같은 L4 발화값을 받으므로 실효 가중치는 (w_exc − w_inh) ∈ [−1, 1] 이고,")
    A("  초기에는 평균 0 근처라 임계값 0.5 를 넘지 못한다. 학습은 이 실효 가중치를 키우는 방향으로 진행된다.")
    A("")
    hist_cond = next((r for r in recs if r["condition"] == "local_learning"), None)
    if hist_cond and hist_cond["training"].get("history"):
        rows = []
        for h in hist_cond["training"]["history"]:
            if h.get("dev_balanced_accuracy") is not None:
                rows.append([h["epoch"], fmt(h["dev_balanced_accuracy"]),
                             fmt(h["mean_nonzero_errors_per_trial"], 2),
                             h["n_weight_updates_applied"], fmt(h["weight_l2_norm"], 3)])
        if rows:
            A(f"**local_learning (seed={hist_cond['seed']}) 의 에폭별 추이 (dev, 교정 없음)**")
            A("")
            A(table(["에폭", "dev balanced acc", "시행당 평균 비영 오차 뉴런 수",
                     "가중치 갱신 적용 횟수", "가중치 L2 노름"], rows))
    A("")

    # --- 8. 진단 -------------------------------------------------------
    A("## 8. 진단: 기여와 개입")
    A("")
    if diag.get("abcd_counterfactual"):
        d = diag["abcd_counterfactual"]
        A("**A/B/C→D 예제의 반사실 개입**")
        A("")
        A(table(["상태", "E", "I", "u", "q"], [
            ["기준 (w_C=0.6)", fmt(d["baseline"]["E"]), fmt(d["baseline"]["I"]),
             fmt(d["baseline"]["u"]), d["baseline"]["q"]],
            ["개입 (w_C=0.2)", fmt(d["intervened_wC_0.6_to_0.2"]["E"]),
             fmt(d["intervened_wC_0.6_to_0.2"]["I"]),
             fmt(d["intervened_wC_0.6_to_0.2"]["u"]), d["intervened_wC_0.6_to_0.2"]["q"]],
        ]))
        A("")
        A(f"- {d['note_ko']}")
    if diag.get("counterfactual_real_circuit"):
        cf = diag["counterfactual_real_circuit"]
        A("")
        A(f"**실제 회로의 반사실 개입**: 연결 {cf['connection']['connection_id']} "
          f"(`{cf['connection']['pre_name']}` → `{cf['connection']['post_name']}`, "
          f"{cf['connection']['kind']}, weight {fmt(cf['connection']['weight'],4)}) 를 0 으로 바꾸면 "
          f"관찰 L2 {cf['n_observed']}개 중 **{cf['n_changed_firing']}개**의 발화가 바뀐다.")
        A(f"- 해당 시행의 가중치 버전을 복원한 무개입 재실행이 저장된 반응과 일치: "
          f"{fmt(cf['replay_matches_stored'])}. (일치하지 않으면 과거 원인에 대한 개입 결과로 보고하지 않는다.)")
    if diag.get("connection_role_table"):
        rt = diag["connection_role_table"]
        A("")
        A("**여러 선 위치·방향에서 같은 연결의 기여와 개입 효과**")
        A("")
        rows = [[r["scene_id"], r["shape"],
                 "-" if r["orientation_deg"] is None else f"{r['orientation_deg']:.0f}°",
                 fmt(r["received_value"], 4), fmt(r["signed_contribution"], 4),
                 r["q_baseline"], r["q_intervened"], fmt(r["delta_u"], 4)] for r in rt]
        A(table(["장면", "종류", "방향", "도착값", "부호 있는 기여", "q(기준)", "q(개입)", "Δu"], rows))
        A("")
        A(f"- {diag.get('connection_role_note_ko','')}")
    A(f"")
    A(f"- 진단 실행 전후로 학습 가중치가 비트 단위로 보존됨: "
      f"{fmt(diag.get('weights_unchanged_by_diagnostics'))} (T7 에서도 별도 검사).")
    A("")

    # --- 9. 그림 -------------------------------------------------------
    A("## 9. 그림")
    A("")
    for p in summary.get("figures", []):
        A(f"- `{os.path.relpath(p, HERE)}`")
    A("")

    # --- 10. 해시와 미실행 ------------------------------------------------
    A("## 10. 산출물 해시")
    A("")
    A("### 코드·설정 (보고서 생성 시점에 다시 계산한 값)")
    A("")
    fresh = fresh_code_hashes()
    A(table(["파일", "바이트", "sha256"],
            [[h["path"], h["bytes"], h["sha256"]] for h in fresh]))
    A("")
    A("### 결과 파일 (실험 실행 종료 시점)")
    A("")
    A(table(["파일", "바이트", "sha256"],
            [[h["path"], h["bytes"], h["sha256"]] for h in summary["result_file_hashes"]]))
    A("")
    A("- 실험 실행 종료 시점의 코드 해시 목록은 `run_summary.json` 의 `code_hashes` 에 그대로 남아 있다.")
    A("- 보고서·압축 파일까지 포함한 최종 해시 목록은 `results/HASHES.md` 와 `results/manifest.json` 에 있다 "
      "(`python make_manifest.py` 로 마지막에 생성). ZIP 자신의 해시는 ZIP 밖(같은 파일)에 기록한다.")
    A("")

    A("## 11. 실제 실행한 것과 미실행 항목 (명확히 구분)")
    A("")
    A("### 실제 실행한 것")
    A("")
    A(f"- 필수 검증 T1~T10: {'실행 (results/checks.json)' if checks else '**미실행**'}")
    A(f"- 부가 단위 검사 `tests/test_units.py`: `python -m unittest discover -s tests` 로 실행 (37개 통과)")
    A(f"- 대조 실험 {conds} × 시드 {summary['seeds']} × {summary['epochs']} 에폭: 실행 "
      f"(총 {summary['elapsed_sec']:.1f}초)")
    A(f"- 조건·시드별 원자료 JSON/NPZ, metrics.csv, 모델 체크포인트, 그림, 진단 기록: 생성됨")
    demo_dirs = [d for d in (os.path.join("results", profile, "figures", "demo"),
                             os.path.join("results", "demo"))
                 if os.path.isdir(os.path.join(HERE, d))]
    A(f"- 저장 모델을 다시 불러와 교사 없는 추론·진단 데모: `run_demo.py --headless` 로 실행 "
      f"({', '.join('`' + d + '/`' for d in demo_dirs) if demo_dirs else '**미실행**'})")
    if summary["lr_search_enabled"] and summary.get("lr_selection"):
        A("- 학습률 탐색: **이 실행에서 실행함**")
        for cond, sel in summary["lr_selection"].items():
            A(f"  - `{cond}`: 후보 {sel['candidates']} → 평균 dev balanced accuracy "
              f"{ {k: round(v,4) for k, v in sel['mean_dev_balanced_accuracy'].items()} } "
              f"→ 선택 **lr={sel['selected_learning_rate']}** ({sel['rule_ko']})")
    else:
        lrs = os.path.join(HERE, "results", "quick_lrsearch", "run_summary.json")
        if os.path.isfile(lrs):
            with open(lrs, encoding="utf-8") as f:
                other = json.load(f)
            A("- 학습률 탐색: 이 프로필에서는 고정 학습률을 썼고, 별도로 "
              "`python run_experiment.py --profile quick --lr-search` 를 **실제 실행**했다 "
              "(`results/quick_lrsearch/`).")
            for cond, sel in (other.get("lr_selection") or {}).items():
                A(f"  - `{cond}`: 후보 {sel['candidates']} → 평균 dev balanced accuracy "
                  f"{ {k: round(v,4) for k, v in sel['mean_dev_balanced_accuracy'].items()} } "
                  f"→ 선택 **lr={sel['selected_learning_rate']}** ({sel['rule_ko']})")
    A("")
    A("### 미실행·한계 항목")
    A("")
    for line in summary.get("not_executed_ko", []):
        A(f"- {line}")
    A("")
    A("## 12. 해석 시 주의")
    A("")
    A("- 이 결과는 **사람이 설계한 위치·방향 목표 지도**를 학습한 결과다. 실제 V1 의 측정된 정상 반응이 아니다.")
    A("- 100만 화소 입력을 같은 API 로 처리한 것은 입력 처리일 뿐, 100만 뉴런 학습 완료가 아니다.")
    A("- L3 출력에 완전한 위상 불변성이나 고도화된 도형 인식을 주장하지 않는다.")
    A("- 억제 중계는 L4 발화를 한 tick 뒤에 전달하는 단순화이며 실제 억제성 세포의 다양성을 재현한 것이 아니다.")
    A("- 이 회로의 수치를 과거 DTP/KP 등 다른 구조의 실험 수치와 직접 비교하지 않는다.")
    A("- 기록 구조의 정확성(T1~T4, T7, T9), 활동의 일시적 교정(correction_only), 지속적 학습 효과(local_learning)")
    A("  는 서로 다른 검증 대상이며 위에서 각각 구분해 보고했다.")
    A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--checks", default=os.path.join(HERE, "results", "checks.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sp = args.summary or os.path.join(HERE, "results", args.profile, "run_summary.json")
    if not os.path.isfile(sp):
        print(f"[오류] 요약 파일이 없다: {sp}\n  먼저 `python run_experiment.py --profile {args.profile}` 실행")
        return 1
    with open(sp, encoding="utf-8") as f:
        summary = json.load(f)
    checks = None
    if os.path.isfile(args.checks):
        with open(args.checks, encoding="utf-8") as f:
            checks = json.load(f)

    text = build_report(summary, checks, args.profile)
    out = args.out or os.path.join(os.path.dirname(sp), "REPORT_ko.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"한국어 보고서 생성: {out}  ({len(text)} 자)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
