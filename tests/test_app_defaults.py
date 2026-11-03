import unittest
from unittest.mock import patch

import numpy as np

import app


class AppDefaultsAndSubmitTests(unittest.TestCase):
    @staticmethod
    def _pipeline_result():
        return {
            "annotated_image": np.zeros((12, 16, 3), dtype=np.uint8),
            "frame_usable": True,
            "frame_quality": {"blur": 50.0, "brightness": 0.4, "contrast": 0.1},
            "plaque_present": True,
            "quantification": {
                "area_px": 50, "area_mm2_est": 0.5,
                "assumed_spacing_mm": 0.1, "calibrated": False,
            },
            "imt": {
                "mean_imt_mm": None, "max_imt_mm": None,
                "mean_imt_px": 12.0, "mean_imt_mm_est": 1.2,
                "demo": True, "measurement_failed": False,
            },
            "imt_class": "unknown",
            "risk": {
                "tier": "unavailable", "score": None,
                "reasons": ["A calibrated IMT measurement is unavailable."],
            },
            "risk_unavailable_reasons": [
                "A calibrated IMT measurement is unavailable; demo estimates are excluded from scoring.",
                "Calibrated plaque area is unavailable; scanner pixel spacing is required.",
            ],
            "human_in_the_loop": True,
        }

    def test_example_defaults_and_blank_scanner_spacing(self):
        self.assertEqual(app.age.get_config()["value"], 65)
        self.assertEqual(app.sbp.get_config()["value"], 130)
        self.assertEqual(app.ldl.get_config()["value"], 110)
        self.assertEqual(app.smoking.get_config()["value"], "No")
        self.assertEqual(app.diabetes.get_config()["value"], "No")
        self.assertIsNone(app.spacing.get_config()["value"])

        for component in (app.age, app.sbp, app.ldl, app.smoking, app.diabetes):
            self.assertIn("example value", component.get_config()["info"].lower())
        self.assertIn(
            "not actual patient data",
            repr([c["props"] for c in app.demo.config["components"]]).lower(),
        )

    def test_submit_wires_defaults_and_returns_analysis_when_risk_unavailable(self):
        callback = next(
            event for event in app.demo.config["dependencies"]
            if event.get("api_name") == "analyze"
        )
        component_by_id = {
            component["id"]: component
            for component in app.demo.config["components"]
        }
        submitted_labels = [
            component_by_id[component_id]["props"].get("label")
            for component_id in callback["inputs"]
        ]
        self.assertEqual(submitted_labels[:7], [
            "Carotid B-mode still", "Age (years)", "Systolic BP (mmHg)",
            "LDL (mg/dL)", "Smoking", "Diabetes", "Pixel spacing (mm/px)",
        ])

        image = np.zeros((12, 16, 3), dtype=np.uint8)
        with patch.object(app.pipe, "run_pipeline", return_value=self._pipeline_result()) as run_pipeline:
            out_image, summary = app.analyze(
                image, 65, 130, 110, "No", "No", None, False)

        kwargs = run_pipeline.call_args.kwargs
        self.assertIsNone(kwargs["pixel_spacing_mm"])
        self.assertEqual(kwargs["clinical"], {
            "age": 65, "sbp": 130, "ldl": 110,
            "smoking": False, "diabetes": False,
        })
        self.assertEqual(kwargs["denoise"], "both")
        self.assertIsNotNone(out_image)
        self.assertIn("Analysis Summary", summary)
        self.assertIn("Risk assessment unavailable", summary)
        self.assertIn("scanner pixel spacing is required", summary.lower())
        self.assertNotIn("demo estimate", summary.lower())

    def test_callback_normalizes_blank_and_zero_spacing_and_preserves_valid_spacing(self):
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        for submitted, expected in ((None, None), ("", None), (0, None), (0.125, 0.125)):
            with self.subTest(submitted=submitted):
                with patch.object(
                    app.pipe, "run_pipeline", return_value=self._pipeline_result()
                ) as run_pipeline:
                    app.analyze(image, 65, 130, 110, "No", "No", submitted, False)
                self.assertEqual(
                    run_pipeline.call_args.kwargs["pixel_spacing_mm"], expected
                )

    def test_invalid_spacing_shows_message_without_calling_pipeline(self):
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        for submitted in (-0.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(submitted=submitted):
                with patch.object(app.pipe, "run_pipeline") as run_pipeline:
                    out_image, summary = app.analyze(
                        image, 65, 130, 110, "No", "No", submitted, False)
                run_pipeline.assert_not_called()
                self.assertIsNone(out_image)
                self.assertIn("Pixel spacing must be finite and greater than 0", summary)


if __name__ == "__main__":
    unittest.main()
