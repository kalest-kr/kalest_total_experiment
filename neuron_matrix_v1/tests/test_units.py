"""tests/test_units.py -- 표준 라이브러리 unittest 단위 검사.

필수 검증 T1~T10 은 ``run_checks.py`` 가 담당한다. 여기서는 그 검사들이
직접 다루지 않는 작은 구성 요소를 확인한다.

실행::

    python -m unittest discover -s tests -v
    python tests/test_units.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import DatasetConfig, SceneSpec, point_segment_distance, render_scene
from src.local_learning import LearningConfig, compute_local_update_for_neuron
from src.metrics import confusion, evaluate_map, per_neuron_metrics, rates_from_confusion
from src.protocols import NotImplementedStage, status_report
from src.retina import RetinaConfig, RetinaEncoder, srgb_to_linear, to_float01
from src.rngs import stream
from src.spatial_mapping import GridConfig, InputNormalizer, bilinear_sample, build_grid
from src.teacher import orientation_difference_deg


class TestRetina(unittest.TestCase):
    def test_to_float01_uint8(self):
        a = np.array([[0, 128, 255]], dtype=np.uint8)
        np.testing.assert_allclose(to_float01(a), [[0.0, 128 / 255, 1.0]])

    def test_to_float01_rejects_out_of_range_float(self):
        with self.assertRaises(ValueError):
            to_float01(np.array([[1.5]], dtype=np.float64))

    def test_srgb_linearization_endpoints(self):
        self.assertAlmostEqual(float(srgb_to_linear(np.array(0.0))), 0.0)
        self.assertAlmostEqual(float(srgb_to_linear(np.array(1.0))), 1.0)
        # 중간 회색은 선형값이 더 작아야 한다
        self.assertLess(float(srgb_to_linear(np.array(0.5))), 0.5)

    def test_gray_from_rgb_uses_rec709(self):
        enc = RetinaEncoder(RetinaConfig())
        img = np.zeros((2, 2, 3))
        img[..., 1] = 1.0  # 순수 녹색
        g = enc.to_gray(img, input_colorspace="linear")
        np.testing.assert_allclose(g, np.full((2, 2), 0.7152))

    def test_gray_requires_explicit_colorspace(self):
        enc = RetinaEncoder(RetinaConfig())
        with self.assertRaises(ValueError):
            enc.to_gray(np.zeros((2, 2)), input_colorspace="unknown")

    def test_dog_uniform_image_is_zero(self):
        enc = RetinaEncoder(RetinaConfig())
        out = enc.encode(np.full((32, 32), 0.5), input_colorspace="linear")
        self.assertLess(float(np.abs(out["dog"]).max()), 1e-12)
        self.assertTrue(np.all(out["on"] >= 0) and np.all(out["off"] >= 0))

    def test_on_off_split_is_complementary(self):
        enc = RetinaEncoder(RetinaConfig())
        img = np.zeros((32, 32)); img[16, :] = 1.0
        out = enc.encode(img, input_colorspace="linear")
        np.testing.assert_allclose(out["on"] - out["off"], out["dog"], atol=1e-15)
        self.assertTrue(np.all(out["on"] * out["off"] == 0.0))


class TestGrid(unittest.TestCase):
    def test_grid_formula(self):
        cfg = GridConfig(radial_bins=8, angular_bins=16, r0_px=2.0)
        g = build_grid(64, 64, cfg)
        R, r0, K = g.R, cfg.r0_px, cfg.radial_bins
        for k in range(K):
            rho = (k + 0.5) / K
            expect = r0 * (np.exp(rho * np.log(1 + R / r0)) - 1.0)
            got = g.r_k[g.radial_bin == k][0]
            self.assertAlmostEqual(float(got), float(expect), places=12)

    def test_sigma_pool_rule(self):
        cfg = GridConfig()
        g = build_grid(64, 64, cfg)
        arc = g.r_k[:: cfg.angular_bins] * 2 * np.pi / cfg.angular_bins
        expect = np.maximum(0.5, 0.5 * np.maximum(g.delta_r, arc))
        np.testing.assert_allclose(g.sigma_by_bin, expect)

    def test_rf_radius_rule(self):
        g = build_grid(64, 64, GridConfig())
        np.testing.assert_allclose(g.rf_radius, np.maximum(2.0, 2.0 * g.sigma_pool))

    def test_points_inside_image_and_fov(self):
        g = build_grid(64, 64, GridConfig())
        self.assertTrue(g.x.min() >= 0 and g.x.max() <= 63)
        self.assertTrue(g.y.min() >= 0 and g.y.max() <= 63)
        d = np.hypot(g.x - g.cx, g.y - g.cy)
        self.assertTrue(np.all(d <= g.R + 1e-9))

    def test_valid_fov_mask_excludes_corners(self):
        g = build_grid(64, 64, GridConfig())
        m = g.valid_fov_mask()
        self.assertFalse(bool(m[0, 0]))
        self.assertTrue(bool(m[32, 32]))

    def test_bilinear_sample_exact_at_integers(self):
        img = np.arange(16, dtype=np.float64).reshape(4, 4)
        v = bilinear_sample(img, np.array([1.0, 2.0]), np.array([0.0, 3.0]))
        np.testing.assert_allclose(v, [1.0, 14.0])

    def test_bilinear_sample_midpoint(self):
        img = np.array([[0.0, 2.0], [4.0, 6.0]])
        v = bilinear_sample(img, np.array([0.5]), np.array([0.5]))
        np.testing.assert_allclose(v, [3.0])


class TestNormalizer(unittest.TestCase):
    def test_percentile_scale(self):
        n = InputNormalizer(95.0).fit(np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0]))
        self.assertFalse(n.degenerate)
        self.assertAlmostEqual(n.scale, float(np.percentile([1, 2, 3, 4], 95)))

    def test_all_zero_is_degenerate_and_safe(self):
        n = InputNormalizer().fit(np.zeros(10))
        self.assertTrue(n.degenerate)
        self.assertEqual(n.scale, 1.0)
        np.testing.assert_allclose(n.transform(np.zeros(3)), np.zeros(3))

    def test_transform_clips_to_unit_range(self):
        n = InputNormalizer().fit(np.array([1.0]))
        out = n.transform(np.array([-5.0, 0.5, 100.0]))
        self.assertTrue(np.all(out >= 0.0) and np.all(out <= 1.0))

    def test_transform_requires_fit(self):
        with self.assertRaises(RuntimeError):
            InputNormalizer().transform(np.zeros(3))


class TestGeometryAndTeacher(unittest.TestCase):
    def test_point_segment_distance_is_finite_segment(self):
        # 무한 직선이면 0 이 되지만 유한 선분이므로 끝점까지의 거리여야 한다
        d = point_segment_distance(np.array([10.0]), np.array([0.0]), (0.0, 0.0), (1.0, 0.0))
        np.testing.assert_allclose(d, [9.0])

    def test_point_segment_distance_perpendicular(self):
        d = point_segment_distance(np.array([0.5]), np.array([3.0]), (0.0, 0.0), (1.0, 0.0))
        np.testing.assert_allclose(d, [3.0])

    def test_orientation_difference_is_180_periodic(self):
        np.testing.assert_allclose(orientation_difference_deg(np.array([0.0]), 180.0), [0.0])
        np.testing.assert_allclose(orientation_difference_deg(np.array([0.0]), 170.0), [10.0])
        np.testing.assert_allclose(orientation_difference_deg(np.array([90.0]), 0.0), [90.0])


class TestDataset(unittest.TestCase):
    def test_blank_scene_is_uniform(self):
        sc = SceneSpec("s", "s", "blank", None, None, 0.0, 0, 0.0, 16, 16)
        img = render_scene(sc)
        self.assertAlmostEqual(float(img.min()), 0.5)
        self.assertAlmostEqual(float(img.max()), 0.5)

    def test_line_polarity(self):
        bright = SceneSpec("b", "b", "horizontal_line", ((2.0, 8.0), (13.0, 8.0)),
                           0.0, 0.3, +1, 1.5, 16, 16)
        dark = SceneSpec("d", "d", "horizontal_line", ((2.0, 8.0), (13.0, 8.0)),
                         0.0, 0.3, -1, 1.5, 16, 16)
        self.assertGreater(float(render_scene(bright).max()), 0.5)
        self.assertLess(float(render_scene(dark).min()), 0.5)

    def test_image_stays_in_unit_range(self):
        sc = SceneSpec("s", "s", "vertical_line", ((8.0, 1.0), (8.0, 14.0)),
                       90.0, 0.45, 1, 2.5, 16, 16)
        img = render_scene(sc)
        self.assertTrue(np.all(img >= 0.0) and np.all(img <= 1.0))

    def test_generate_scenes_deterministic_per_seed(self):
        from src.datasets import generate_scenes
        a = generate_scenes(20, stream(7, "data"), DatasetConfig(), "x")
        b = generate_scenes(20, stream(7, "data"), DatasetConfig(), "x")
        self.assertEqual([s.to_dict() for s in a], [s.to_dict() for s in b])


class TestRngStreams(unittest.TestCase):
    def test_streams_are_independent_and_reproducible(self):
        a1 = stream(42, "data").random(5)
        a2 = stream(42, "data").random(5)
        b = stream(42, "init").random(5)
        np.testing.assert_array_equal(a1, a2)
        self.assertFalse(np.array_equal(a1, b))

    def test_unknown_stream_rejected(self):
        with self.assertRaises(KeyError):
            stream(42, "not_registered")


class TestLocalRule(unittest.TestCase):
    def test_pure_function_signs(self):
        cfg = LearningConfig(epsilon=1e-8, learning_rate=0.1, weight_max=1.0)
        e = np.array([1.0, 1.0, 0.0])
        sign = np.array([+1.0, -1.0, +1.0])
        tr = np.array([True, True, True])
        G = compute_local_update_for_neuron(+1.0, e, sign, tr, cfg)
        self.assertGreater(G[0], 0.0)     # 흥분 강화
        self.assertLess(G[1], 0.0)        # 억제 약화
        self.assertEqual(G[2], 0.0)       # 입력 0 -> 갱신 0

    def test_zero_error_gives_zero_update(self):
        cfg = LearningConfig()
        G = compute_local_update_for_neuron(
            0.0, np.array([1.0, 0.5]), np.array([1.0, -1.0]), np.array([True, True]), cfg)
        np.testing.assert_array_equal(G, np.zeros(2))

    def test_non_trainable_inputs_never_updated(self):
        cfg = LearningConfig()
        G = compute_local_update_for_neuron(
            1.0, np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([True, False]), cfg)
        self.assertEqual(G[1], 0.0)

    def test_normalisation_makes_delta_u_equal_lr(self):
        # Z = sum e^2 이면 u 변화량은 대략 learning_rate * d 가 된다
        cfg = LearningConfig(epsilon=0.0, learning_rate=0.05, weight_max=1.0)
        e = np.array([1.0, 1.0, 1.0, 1.0])
        sign = np.ones(4)
        G = compute_local_update_for_neuron(1.0, e, sign, np.ones(4, bool), cfg)
        self.assertAlmostEqual(float((G * e).sum()), 0.05, places=12)


class TestMetrics(unittest.TestCase):
    def test_confusion_counts(self):
        c = confusion(np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]))
        self.assertEqual(c, {"TP": 1, "FP": 1, "TN": 1, "FN": 1})

    def test_balanced_accuracy_undefined_without_positives(self):
        r = rates_from_confusion({"TP": 0, "FP": 1, "TN": 3, "FN": 0})
        self.assertIsNone(r["balanced_accuracy"])
        self.assertIsNone(r["recall_tpr"])

    def test_perfect_prediction(self):
        y = np.array([[1, 0], [0, 1]])
        m = evaluate_map(y, y)
        self.assertAlmostEqual(m["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)

    def test_per_neuron_undefined_is_nan_not_zero(self):
        y = np.array([[0, 1], [0, 1]])   # 첫 뉴런은 양성이 전혀 없다
        p = np.array([[0, 1], [0, 1]])
        m = per_neuron_metrics(y, p)
        self.assertTrue(np.isnan(m["balanced_accuracy"][0]))
        self.assertEqual(m["n_undefined"], 2)
        self.assertEqual(m["n_no_positive_target"], 1)


class TestProtocols(unittest.TestCase):
    def test_unimplemented_areas_raise(self):
        for area in ("V2", "V4", "IT"):
            with self.assertRaises(NotImplementedError):
                NotImplementedStage(area).forward(np.zeros(3))

    def test_status_report_lists_scope(self):
        r = status_report()
        self.assertEqual(r["implemented_areas"], ["V1"])
        self.assertIn("V2", r["not_implemented_areas"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
