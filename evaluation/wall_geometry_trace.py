"""Compare checkpoint-compatible stretch with a letterbox alternative.

Uses three existing CUBS held-out images and their wall masks/calibrated IMT
references. This is a focused inference trace, not training or a full benchmark.
Run from the project root: python evaluation/wall_geometry_trace.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from tiartery.geometry import normalize_resize_mode, resize_input, restore_probability
from tiartery.imt import imt_from_wall_mask, interfaces_from_wall_mask
from tiartery.segmentation import load_model


DATA = PROJECT.parent / "data" / "cubs"
OUT = PROJECT / "evaluation" / "wall_geometry_trace_20261030"
SAMPLES = ("083", "086", "062")


def infer(model, gray, mode):
    mode = normalize_resize_mode(mode)
    small = resize_input(gray, model.input_size, mode)
    x = torch.from_numpy(small[None, None]).float()
    with torch.no_grad():
        prob_small = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    prob = restore_probability(prob_small, gray.shape, mode)
    mask = prob >= model.threshold
    return mask, prob


def boundary_errors(pred, prob, gt, calibration):
    pred_li, pred_ma = interfaces_from_wall_mask(pred.astype(np.uint8), prob=prob)
    gt_li, gt_ma = interfaces_from_wall_mask(gt.astype(np.uint8), subpixel=False)
    gt_li_by_x = {int(x): y for x, y in gt_li}
    gt_ma_by_x = {int(x): y for x, y in gt_ma}
    common = [
        (li, ma, gt_li_by_x[int(li[0])], gt_ma_by_x[int(ma[0])])
        for li, ma in zip(pred_li, pred_ma)
        if int(li[0]) in gt_li_by_x and int(ma[0]) in gt_ma_by_x
    ]
    if not common:
        return {"columns_compared": 0, "li_ma_boundary_mae_px": None,
                "li_ma_boundary_mae_mm": None}
    errors = [abs(li[1] - gt_li_y) for li, _, gt_li_y, _ in common]
    errors.extend(abs(ma[1] - gt_ma_y) for _, ma, _, gt_ma_y in common)
    mae_px = float(np.mean(errors))
    return {"columns_compared": len(common),
            "li_ma_boundary_mae_px": mae_px,
            "li_ma_boundary_mae_mm": mae_px * calibration}


def draw_boundaries(gray, pred, prob, gt, output):
    canvas = cv2.cvtColor((np.clip(gray, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    gt_li, gt_ma = interfaces_from_wall_mask(gt.astype(np.uint8), subpixel=False)
    pred_li, pred_ma = interfaces_from_wall_mask(pred.astype(np.uint8), prob=prob)

    def line(points, color):
        if len(points) > 1:
            # Points are already OpenCV (x, y); reversing them transposed the
            # displayed LI/MA traces and made the earlier overlays misleading.
            pts = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)

    # Ground truth LI/MA = green/yellow; predicted LI/MA = red/magenta.
    line(gt_li, (0, 255, 0))
    line(gt_ma, (0, 220, 255))
    line(pred_li, (0, 0, 255))
    line(pred_ma, (255, 0, 255))
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1] - 1, 430), 30), (0, 0, 0), -1)
    cv2.putText(canvas, "GT LI/MA: green/yellow  Pred LI/MA: red/magenta",
                (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text())
    model = load_model(str(PROJECT / "models" / "wall_unet.pt"), "cpu")
    modes = {"checkpoint_compatible_stretch": "stretch",
             "letterbox_alternative": "letterbox"}
    report = {
        "checkpoint": "models/wall_unet.pt",
        "checkpoint_resize_mode_metadata": None,
        "loader_mode_after_fix": model.resize_mode,
        "note": "Checkpoint omits resize_mode. Stretch exactly reproduces its stored val_dice on the validation split; the training command/config does not record the actual mode.",
        "threshold": model.threshold,
        "samples": {},
    }

    for stem in SAMPLES:
        image_path = DATA / "test" / f"{stem}.png"
        mask_path = DATA / "test" / f"{stem}_wall.png"
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        meta = manifest[f"test/{stem}"]
        calibration = float(meta["cf_mm_px"])
        gt_mask_imt = imt_from_wall_mask(gt.astype(np.uint8) * 255,
                                         calibration, subpixel=False)
        rows = {}

        for name, mode in modes.items():
            pred, prob = infer(model, gray, mode)
            intersection = int(np.logical_and(pred, gt).sum())
            dice = 2 * intersection / max(int(pred.sum() + gt.sum()), 1)
            union = int(np.logical_or(pred, gt).sum())
            iou = intersection / max(union, 1)
            imt = imt_from_wall_mask(pred.astype(np.uint8) * 255,
                                     calibration, prob=prob)
            imt_mm = imt.get("mean_imt_mm")
            reference_mm = float(meta["gt_imt_mm"])
            row = {
                "resize_mode": mode,
                "image_shape_hw": list(gray.shape),
                "pred_wall_pixels": int(pred.sum()),
                "gt_wall_pixels": int(gt.sum()),
                "dice": float(dice),
                "iou": float(iou),
                "li_ma_boundary_error": boundary_errors(pred, prob, gt, calibration),
                "mean_imt_px": imt.get("mean_imt_px"),
                "mean_imt_mm": imt_mm,
                "reference_mask_imt_px": gt_mask_imt.get("mean_imt_px"),
                "reference_mask_imt_mm": gt_mask_imt.get("mean_imt_mm"),
                "reference_gt_imt_mm": reference_mm,
                "imt_absolute_error_mm": (abs(float(imt_mm) - reference_mm)
                                          if imt_mm is not None else None),
                "pixel_spacing_mm_px": calibration,
            }
            rows[name] = row
            draw_boundaries(gray, pred, prob, gt,
                            OUT / f"{stem}_{name}_boundaries.png")
        report["samples"][stem] = {
            "source_id": meta["src"],
            "checkpoint_compatible_stretch": rows["checkpoint_compatible_stretch"],
            "letterbox_alternative": rows["letterbox_alternative"],
        }

    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Overlays and metrics saved in {OUT}")


if __name__ == "__main__":
    main()
