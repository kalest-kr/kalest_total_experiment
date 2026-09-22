"""해부 구조·배선·자극·분할 테스트 (명세 6, 7, 11절)."""

from __future__ import annotations

import numpy as np

from cortex import stimuli as S
from cortex.anatomy import layer_summary, ocular_dominance_map, orientation_map
from cortex.areas import connectivity_summary


def test_all_declared_layers_have_neurons(minimal_cfg, minimal_model):
    counts = layer_summary(minimal_model.anat)["counts"]
    present = {k.rsplit("/", 1)[0] for k in counts}
    for aname, a in minimal_cfg["anatomy"]["areas"].items():
        if a["kind"] == "retina":
            assert f"{aname}/L_input" in present
            continue
        for lname, n in a["neurons_per_layer"].items():
            if int(n) > 0:
                assert f"{aname}/{lname}" in present


def test_every_enabled_rule_made_synapses(minimal_model):
    empty = [r["name"] for r in minimal_model.wiring_report.per_rule
             if r.get("enabled") and r.get("n", 0) == 0]
    assert not empty, f"이름만 존재하는 경로: {empty}"


def test_delays_are_at_least_one_step(minimal_model):
    assert int(minimal_model.table.effective_delay_steps.min()) >= 1


def test_positions_are_three_dimensional(minimal_model):
    p = minimal_model.anat.population.arrays.position_mm
    assert p.shape[1] == 3
    assert np.isfinite(p).all()
    assert p[:, 2].min() <= 0.0, "깊이는 표면(z=0) 아래로 간다"


def test_orientation_map_is_in_range():
    rng = np.random.default_rng(0)
    u, v = np.meshgrid(np.linspace(0, 2, 20), np.linspace(0, 2, 20))
    theta = orientation_map(u.ravel(), v.ravel(), 0.8, rng)
    assert theta.min() >= 0.0 and theta.max() < np.pi


def test_ocular_dominance_is_separate_property():
    u = np.linspace(0, 2, 50)
    od = ocular_dominance_map(u, 0.4, 0.6)
    assert od.min() >= -0.6 - 1e-9 and od.max() <= 0.6 + 1e-9
    assert od.std() > 0.0


def test_connectivity_summary_has_pathways(minimal_cfg, minimal_model):
    s = connectivity_summary(minimal_cfg, minimal_model.anat, minimal_model.table)
    assert s["by_pathway"], "경로별 시냅스 집계가 있어야 한다"


def test_stimulus_images_are_in_range():
    img = S.bar_image(32, 32, 16, 16, 20, 3, 0.5)
    assert img.shape == (32, 32, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_uniform_image_is_constant():
    img = S.uniform_image(16, 16, 0.5)
    assert float(img.std()) == 0.0


def test_splits_do_not_leak_variants(minimal_cfg):
    cfg = dict(minimal_cfg)
    cfg["experiment"] = dict(cfg["experiment"])
    cfg["experiment"]["stimuli"] = [{
        "kind": "shape_classes",
        "params": {"size_px": 32, "classes": ["circle", "square"],
                   "n_per_class": 6, "n_variants": 3, "radius_px": 6.0}}]
    stims = S.generate(cfg, np.random.default_rng(0))
    splits = S.split_stimuli(stims, cfg, np.random.default_rng(1))
    rep = S.split_report(splits)
    assert rep["no_overlap"], rep["overlaps"]


def test_binocular_zero_disparity_is_flagged():
    cfg = {"retina": {"image": {"max_side_px": 32}},
           "experiment": {"stimuli": [{"kind": "binocular",
                                       "params": {"size_px": 32,
                                                  "disparity_px": [0.0]}}]}}
    stims = S.generate(cfg, np.random.default_rng(0))
    assert "복제" in stims[0].params["note_ko"]
