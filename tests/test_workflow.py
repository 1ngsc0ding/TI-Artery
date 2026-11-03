import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class SplitTests(unittest.TestCase):
    def setup_data(self, root):
        for split in ['train', 'test']:
            (root/split).mkdir()
        for i in range(10):
            split = 'train' if i < 8 else 'test'
            (root/split/f'{i}.png').write_bytes(f'image-{i}'.encode())
            (root/split/f'{i}_labeled.png').write_bytes(b'mask')

    def run_split(self, root, out, *args):
        return subprocess.run([sys.executable, str(ROOT/'prepare_plaque_validation.py'),
            '--source', str(root), '--out', str(out), *args], capture_output=True, text=True)

    def test_requires_explicit_independence_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'data'; root.mkdir(); self.setup_data(root)
            r = self.run_split(root, Path(tmp)/'out')
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse((Path(tmp)/'out').exists())

    def test_group_split_and_originals_preserved(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'data'; root.mkdir(); self.setup_data(root)
            groups = Path(tmp)/'groups.csv'
            with groups.open('w', newline='') as f:
                w=csv.writer(f); w.writerow(['filename','patient_id'])
                for i in range(10):
                    w.writerow([f'{"train" if i<8 else "test"}/{i}.png', f'P{i//2}'])
            out=Path(tmp)/'out'
            r=self.run_split(root,out,'--groups',str(groups))
            self.assertEqual(r.returncode,0,r.stderr)
            items=json.loads((out/'split_manifest.json').read_text())['items']
            assignments={}
            for x in items:
                assignments.setdefault(x['patient_id'],set()).add(x['split'])
            self.assertTrue(all(len(x)==1 for x in assignments.values()))
            self.assertEqual(len(list(root.rglob('*.png'))),20)

    def test_rejects_duplicate_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data'; root.mkdir(); self.setup_data(root)
            (root/'test/8.png').write_bytes((root/'train/0.png').read_bytes())
            r=self.run_split(root,Path(tmp)/'out','--allow-image-split')
            self.assertNotEqual(r.returncode,0)
            self.assertIn('Duplicate image',r.stderr)

class PipelineUnitTests(unittest.TestCase):
    def test_srad_preserves_shape(self):
        import numpy as np
        from tiartery.preprocessing import srad, lee_filter
        x = np.random.RandomState(0).rand(32, 40).astype(np.float32)
        y = srad(x, niter=3)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.isfinite(y).all())
        z = lee_filter(x)
        self.assertEqual(z.shape, x.shape)

    def test_imt_from_wall_mask(self):
        import numpy as np
        from tiartery.imt import imt_from_wall_mask, classify_imt
        mask = np.zeros((40, 50), np.uint8)
        mask[10:16, 5:45] = 255  # 6 pixels high -> 5 px gap between first/last
        r = imt_from_wall_mask(mask, pixel_spacing_mm=0.1, subpixel=False)
        self.assertFalse(r["measurement_failed"])
        self.assertAlmostEqual(r["mean_imt_mm"], 0.5, places=5)
        self.assertEqual(classify_imt(0.8), "normal range")
        self.assertEqual(classify_imt(1.2), "thickened")

    def test_planimetry_calibration(self):
        import numpy as np
        from tiartery.plaque import quantify
        mask = np.zeros((10, 10), np.uint8)
        mask[:2, :5] = 255
        q = quantify(mask, 0.2)
        self.assertEqual(q["area_px"], 10)
        self.assertAlmostEqual(q["area_mm2"], 0.4)

class ResultDisplayTests(unittest.TestCase):
    def _base(self, **overrides):
        res = {
            "source": {"source": "image", "n_frames": 1},
            "frame_usable": True,
            "frame_quality": {"blur": 80.0, "brightness": 0.4, "contrast": 0.12},
            "plaque_present": True,
            "quantification": {"area_px": 120, "area_mm2": 4.8, "calibrated": True},
            "imt": {"mean_imt_mm": 0.82, "max_imt_mm": 0.91, "measurement_failed": False,
                    "note": "Geometry from far-wall LI/MA"},
            "imt_class": "normal range",
            "risk": {"tier": "elevated", "score": 2, "reasons": ["plaque detected on B-mode"]},
            "risk_unavailable_reasons": [],
            "human_in_the_loop": True,
        }
        res.update(overrides)
        return res

    def test_summary_card_keeps_overlay_legend_and_technical_log(self):
        from tiartery.report import format_view
        view = format_view(self._base(), clinician_reviewed=False)
        self.assertIn("Analysis Summary", view["summary"])
        self.assertIn("IMT", view["summary"])
        self.assertIn("green = plaque", view["legend"])
        self.assertIn("orange = intima-media wall", view["legend"])
        self.assertIn("blur", view["technical"])

    def test_missing_measurements_use_actual_reason(self):
        from tiartery.report import format_view
        view = format_view(self._base(
            plaque_present=None, quantification=None,
            imt={"mean_imt_mm": None, "max_imt_mm": None, "measurement_failed": True,
                 "note": "Wall model unavailable"},
            imt_class="unknown",
            risk={"tier": "unavailable", "score": None,
                  "reasons": ["Wall model unavailable"]},
            risk_unavailable_reasons=["Wall model unavailable", "Age was not provided."],
        ))
        self.assertIn("Plaque model not loaded", view["summary"])
        self.assertIn("Wall model unavailable", view["summary"])
        self.assertIn("Age was not provided", view["summary"])
        self.assertIn("Risk assessment unavailable", view["summary"])

    def test_no_plaque_does_not_imply_no_disease(self):
        from tiartery.report import summary_card
        res = self._base(plaque_present=False,
                         quantification={"area_px": 0, "calibrated": False,
                                         "note": "uncalibrated: pixel spacing unknown"})
        rendered = summary_card(res).lower()
        self.assertIn("no plaque segmented on this frame", rendered)
        self.assertNotIn("disease-free", rendered)
        self.assertNotIn("healthy", rendered)
        self.assertNotIn("rules out disease", rendered)

    def test_uncalibrated_area_explains_spacing(self):
        from tiartery.report import summary_card
        res = self._base(
            quantification={"area_px": 90, "length_px": 30, "thickness_px": 5,
                            "length_mm_est": 3.0, "thickness_mm_est": 0.5,
                            "assumed_spacing_mm": 0.1, "calibrated": False,
                            "note": "uncalibrated: pixel spacing unknown, areas are in px^2 not mm^2"},
            imt={"mean_imt_mm": None, "mean_imt_px": 6.2, "max_imt_px": 7.1,
                 "mean_imt_mm_est": 0.62, "assumed_spacing_mm": 0.1,
                 "measurement_failed": False, "note": "Geometry from far-wall LI/MA"},
            imt_class="unknown",
            risk={"tier": "unavailable", "score": None, "reasons": []},
            risk_unavailable_reasons=[
                "A calibrated IMT measurement is unavailable; demo estimates are excluded from scoring.",
                "Calibrated plaque area is unavailable; scanner pixel spacing is required.",
            ],
        )
        rendered = summary_card(res).lower()
        self.assertIn("uncalibrated plaque measurements", rendered)
        self.assertIn("length 30.0 px", rendered)
        self.assertIn("thickness 5.0 px", rendered)
        self.assertIn("area 90 px²", rendered)
        self.assertIn("uncalibrated imt", rendered)
        self.assertIn("risk assessment unavailable", rendered)
        self.assertIn("scanner pixel spacing is required for calibrated imt and plaque area", rendered)
        self.assertNotIn("assumed pixel spacing", rendered)
        self.assertNotIn("demo estimate", rendered)
        self.assertNotIn("3.0 mm", rendered)
        self.assertNotIn("0.5 mm", rendered)
        self.assertNotIn("0.62 mm", rendered)
        self.assertNotIn("mm²", rendered)
        self.assertNotIn("extensive plaque", rendered)
        self.assertNotIn("refer for specialist evaluation", rendered)

    def test_calibrated_plaque_keeps_size_classification_and_recommendation(self):
        from tiartery.report import summary_card
        rendered = summary_card(self._base(quantification={
            "area_px": 900, "area_mm2": 36.0,
            "length_mm": 25.0, "thickness_mm": 5.0,
            "calibrated": True,
        })).lower()
        self.assertIn("25.0 mm — extensive plaque", rendered)
        self.assertIn("refer for specialist evaluation", rendered)
        self.assertNotIn("imt 0.62 mm", rendered)

    def test_quality_failure_uses_blur_reason(self):
        from tiartery.report import summary_card
        html = summary_card(self._base(
            frame_usable=False,
            frame_quality={"blur": 8.0, "brightness": 0.4, "contrast": 0.12}))
        self.assertIn("image quality check failed", html.lower())

    def test_pipeline_suppresses_risk_without_models_or_spacing(self):
        import numpy as np
        import cv2
        from tiartery.pipeline import run_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "frame.png")
            cv2.imwrite(path, np.full((64, 80), 80, np.uint8))
            res = run_pipeline(path, model=None, wall_model=None,
                               pixel_spacing_mm=None,
                               clinical={"age": 65, "sbp": 130, "ldl": 110})
            self.assertEqual(res["risk"]["tier"], "unavailable")
            self.assertIsNone(res["risk"]["score"])
            joined = " ".join(res["risk_unavailable_reasons"])
            self.assertIn("plaque model", joined.lower())
            self.assertTrue("Wall model unavailable" in joined or "IMT" in joined)

    def test_overlay_legend_does_not_change_mask_pixels(self):
        import numpy as np
        from tiartery.plaque import overlay_mask, draw_overlay_legend
        gray = np.full((40, 60), 0.4, np.float32)
        mask = np.zeros((40, 60), np.uint8)
        mask[20:30, 10:40] = 255
        over = overlay_mask(gray, mask, color=(0, 255, 0))
        labeled = draw_overlay_legend(over)
        self.assertEqual(labeled.shape[0], over.shape[0] + 36)
        self.assertTrue((over == labeled[36:]).all())


class GeometryTests(unittest.TestCase):
    def test_letterbox_roundtrip(self):
        import numpy as np
        from tiartery.geometry import resize_input, restore_probability
        a=np.ones((60,80),dtype=np.float32)
        b=resize_input(a,320,'letterbox')
        self.assertEqual(b.shape,(320,320))
        self.assertTrue(np.all(b[:40]==0))
        np.testing.assert_allclose(restore_probability(b,a.shape,'letterbox'),a)

    def test_missing_resize_mode_uses_matching_stretch_restore(self):
        import numpy as np
        from tiartery.geometry import resize_input, restore_probability
        image = np.ones((60, 80), dtype=np.float32)
        model_input = resize_input(image, 320, None)
        self.assertTrue(np.all(model_input == 1))
        restored = restore_probability(model_input, image.shape, None)
        np.testing.assert_allclose(restored, image)

    def test_predict_prob_pairs_transforms_when_model_mode_is_none(self):
        from unittest.mock import patch
        import numpy as np
        import torch
        from tiartery import segmentation

        class StubModel(torch.nn.Module):
            input_size = 16
            resize_mode = None

            def forward(self, x):
                return torch.zeros_like(x)

        image = np.ones((12, 20), dtype=np.float32)
        with patch("tiartery.segmentation.resize_input",
                   wraps=segmentation.resize_input) as resize, \
             patch("tiartery.segmentation.restore_probability",
                   wraps=segmentation.restore_probability) as restore:
            segmentation.predict_prob(StubModel(), image)

        self.assertEqual(resize.call_args.args[2], "stretch")
        self.assertEqual(restore.call_args.args[2], "stretch")

if __name__=='__main__': unittest.main()
