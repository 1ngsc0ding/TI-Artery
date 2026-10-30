import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from tiartery.pipeline import run_pipeline


class RiskMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        h, w = 240, 320
        y, x = np.mgrid[:h, :w]
        gray = (0.35 + 0.22 * np.sin(x / 13) + 0.18 * np.cos(y / 17))
        gray += np.random.default_rng(7).normal(0, 0.04, gray.shape)
        gray = gray.clip(0, 1)
        self.path = str(Path(self.temp.name) / "frame.png")
        cv2.imwrite(self.path, (gray * 255).astype(np.uint8))

        self.plaque_mask = np.zeros((h, w), dtype=np.uint8)
        self.plaque_mask[80:100, 90:130] = 255
        self.wall_mask = np.zeros((h, w), dtype=np.uint8)
        self.wall_mask[120:132, 30:290] = 255
        self.prob = (self.wall_mask > 0).astype(np.float32)
        self.clinical = {
            "age": 65, "sbp": 130, "ldl": 110,
            "smoking": False, "diabetes": False,
        }

    def _predict(self, model, gray, **kwargs):
        if model == "plaque":
            return self.plaque_mask
        if kwargs.get("return_prob"):
            return self.wall_mask, self.prob
        return self.wall_mask

    def _run(self, spacing):
        with patch("tiartery.segmentation.predict_mask", side_effect=self._predict):
            return run_pipeline(
                self.path, model="plaque", wall_model="wall",
                pixel_spacing_mm=spacing, clinical=self.clinical, denoise="lee")

    def test_blank_spacing_keeps_imt_in_pixels_and_blocks_tier(self):
        result = self._run(None)
        imt = result["imt"]
        self.assertTrue(imt["demo"])
        self.assertIsNone(imt["mean_imt_mm"])
        self.assertIsNone(imt["max_imt_mm"])
        self.assertAlmostEqual(imt["mean_imt_mm_est"], 1.2, places=3)
        self.assertAlmostEqual(imt["mean_imt_px"], 12.0, places=3)
        self.assertIsNone(result["quantification"].get("area_mm2"))
        self.assertIsNone(result["risk"]["score"])
        self.assertEqual(result["risk"]["tier"], "unavailable")
        self.assertTrue(any("calibrated IMT" in r for r in result["risk"]["reasons"]))

    def test_zero_spacing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self._run(0)

    def test_demo_imt_is_excluded_even_with_scanner_spacing(self):
        def demo(gray, pixel_spacing_mm):
            return {"mean_imt_mm": 1.2, "max_imt_mm": 1.4, "gap_px": 12,
                    "gap_px_p90": 14, "demo": True, "measurement_failed": False}

        with patch("tiartery.segmentation.predict_mask", side_effect=self._predict), \
             patch("tiartery.pipeline.imt_mod.estimate_imt_demo", side_effect=demo):
            result = run_pipeline(
                self.path, model="plaque", pixel_spacing_mm=0.1,
                clinical=self.clinical, denoise="lee")
        self.assertIsNone(result["imt"]["mean_imt_mm"])
        self.assertEqual(result["imt"]["mean_imt_mm_est"], 1.2)
        self.assertIsNone(result["risk"]["score"])
        self.assertTrue(any("demo estimates are excluded" in r for r in result["risk"]["reasons"]))

    def test_calibrated_wall_imt_still_contributes_point(self):
        result = self._run(0.1)
        self.assertFalse(result["imt"]["demo"])
        self.assertAlmostEqual(result["imt"]["mean_imt_mm"], 1.2, places=3)
        self.assertEqual(result["quantification"]["area_mm2"], 8.0)
        # Plaque +2, calibrated IMT +1, age +1; area and remaining factors +0.
        self.assertEqual(result["risk"]["score"], 4)
        self.assertEqual(result["risk"]["tier"], "high")
        self.assertTrue(any(r.startswith("IMT ") for r in result["risk"]["reasons"]))

    def test_incomplete_patient_factors_block_complete_tier(self):
        with patch("tiartery.segmentation.predict_mask", side_effect=self._predict):
            result = run_pipeline(
                self.path, model="plaque", wall_model="wall",
                pixel_spacing_mm=0.1, clinical={"age": None, "sbp": 130,
                                                "ldl": 110, "smoking": None,
                                                "diabetes": False},
                denoise="lee")
        self.assertEqual(result["risk"]["tier"], "unavailable")
        self.assertIsNone(result["risk"]["score"])
        self.assertTrue(any("Age was not provided" in r for r in result["risk"]["reasons"]))
        self.assertTrue(any("Smoking status was not provided" in r
                            for r in result["risk"]["reasons"]))

    def test_report_labels_score_and_unavailable_state(self):
        from tiartery.report import format_view

        unavailable = format_view({
            "risk": {"tier": "unavailable", "score": None,
                     "reasons": ["A calibrated IMT measurement is unavailable."]},
            "risk_unavailable_reasons": ["A calibrated IMT measurement is unavailable."],
            "imt": {"mean_imt_mm": None, "mean_imt_px": 12.0,
                    "mean_imt_mm_est": 1.2, "demo": True},
            "quantification": {}, "plaque_present": None,
            "frame_usable": True, "human_in_the_loop": True,
        })["summary"]
        self.assertIn("Risk assessment unavailable", unavailable)
        self.assertIn("not used for scoring", unavailable)

        scored = format_view({
            "risk": {"tier": "high", "score": 4, "reasons": []},
            "risk_unavailable_reasons": [], "imt": {"mean_imt_mm": 1.0},
            "quantification": {}, "plaque_present": True,
            "frame_usable": True, "human_in_the_loop": True,
        })["summary"]
        self.assertIn("Unvalidated rule-based screening score", scored)
        self.assertIn("4 out of 8 possible points", scored)
        self.assertNotIn("/ 8 factors considered", scored)


if __name__ == "__main__":
    unittest.main()
