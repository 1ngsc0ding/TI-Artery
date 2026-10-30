"""Reproduce stored validation Dice and evaluate available held-out masks.

Run from the project root with: python evaluation/run_accuracy_eval.py
Overlays and metrics are written beside this file in accuracy_20261030/.
"""
import hashlib
import json
import sys
import gc
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from tiartery.geometry import resize_input, restore_probability
from tiartery.imt import imt_from_wall_mask
from tiartery.segmentation import load_model


DATA = PROJECT.parent / "data"
OUT = Path(__file__).resolve().parent / "accuracy_20261030"
BATCH = 8


def metric(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    inter = int(np.logical_and(pred, gt).sum())
    ps, gs = int(pred.sum()), int(gt.sum())
    return (1.0 if ps + gs == 0 else 2 * inter / (ps + gs),
            1.0 if ps + gs - inter == 0 else inter / (ps + gs - inter))


def global_metric(rows):
    tp = sum(int(np.logical_and(p, g).sum()) for p, g in rows)
    fp = sum(int(np.logical_and(p, ~g).sum()) for p, g in rows)
    fn = sum(int(np.logical_and(~p, g).sum()) for p, g in rows)
    return {"dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1)}


def read_pairs(dataset, split, suffix):
    folder = dataset / split
    pairs = []
    for ip in sorted(folder.glob("*.png")):
        if ip.stem.endswith(suffix.removesuffix(".png")):
            continue
        mp = ip.with_name(ip.stem + suffix)
        if mp.exists():
            img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
            pairs.append((ip, img, gt))
    return pairs


def infer(model, pairs, training_resolution=False):
    mode = getattr(model, "resize_mode", "stretch")
    size = getattr(model, "input_size", 320)
    results = []
    for start in range(0, len(pairs), BATCH):
        batch = pairs[start:start + BATCH]
        xs = np.stack([resize_input(img, size, mode) for _, img, _ in batch])
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(xs[:, None]).float()))[:, 0].numpy()
        for (ip, img, gt), prob_small in zip(batch, probs):
            if training_resolution:
                gt_eval = resize_input(gt.astype(np.float32), size, mode, mask=True) > 0.5
                prob = prob_small
            else:
                gt_eval = gt
                # Match the production preprocessing and restore path exactly.
                prob = restore_probability(prob_small, img.shape, mode)
            pred = prob > 0.5
            d, i = metric(pred, gt_eval)
            results.append({"path": ip, "img": img, "gt": gt_eval,
                            "prob": prob, "pred": pred, "dice": float(d),
                            "iou": float(i)})
        if (start // BATCH + 1) % 5 == 0 or start + BATCH >= len(pairs):
            print(f"  inferred {min(start + BATCH, len(pairs))}/{len(pairs)}", flush=True)
    return results


def write_overlay(row, name):
    rgb = cv2.cvtColor((np.clip(row["img"], 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rgb[row["gt"]] = (0, 255, 0)  # reference only
    rgb[row["pred"] & ~row["gt"]] = (0, 0, 255)  # false positive
    rgb[~row["pred"] & row["gt"]] = (255, 0, 0)  # false negative
    cv2.imwrite(str(OUT / name), rgb)


def summarize_seg(rows):
    gm = global_metric([(r["pred"], r["gt"]) for r in rows])
    gm.update({"n": len(rows),
               "mean_image_dice": float(np.mean([r["dice"] for r in rows])),
               "mean_image_iou": float(np.mean([r["iou"] for r in rows]))})
    return gm


def split_checks():
    cubs = json.loads((DATA / "cubs/manifest.json").read_text())
    src = {s: {v["src"] for k, v in cubs.items() if k.startswith(s + "/")}
           for s in ("train", "val", "test")}
    hashes = {}
    for s in src:
        hashes[s] = {hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (DATA / "cubs" / s).glob("*.png") if "_wall" not in p.name}
    plaque = json.loads((DATA / "plaque_v3/split_manifest.json").read_text())
    plaque_hashes = {s: {x["image_sha256"] for x in plaque["items"] if x["split"] == s}
                     for s in ("train", "val", "test")}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {
        "cubs_source_id_intersections": {f"{a}/{b}": len(src[a] & src[b]) for a, b in pairs},
        "cubs_exact_image_hash_intersections": {f"{a}/{b}": len(hashes[a] & hashes[b]) for a, b in pairs},
        "plaque_v3_exact_image_hash_intersections": {f"{a}/{b}": len(plaque_hashes[a] & plaque_hashes[b]) for a, b in pairs},
        "plaque_v3_patient_grouped": plaque.get("patient_grouped"),
        "plaque_v3_patient_id_missing_count": sum(x.get("patient_id") is None for x in plaque["items"]),
        "plaque_v3_test_status": plaque.get("test_status"),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    output = {"split_checks": split_checks()}

    # Plaque checkpoint validation metric at training resolution (letterbox,
    # global pixel Dice), then native-size app inference on the manifest test.
    plaque_path = PROJECT / "models/plaque_finetune.pt"
    plaque_state = torch.load(plaque_path, map_location="cpu", weights_only=False)
    plaque_model = load_model(str(plaque_path), "cpu")
    plaque_data = DATA / "plaque_v3"
    plaque_val = infer(plaque_model, read_pairs(plaque_data, "val", "_labeled.png"), True)
    plaque_test = infer(plaque_model, read_pairs(plaque_data, "test", "_labeled.png"))
    pos = [r for r in plaque_test if r["gt"].any()]
    neg = [r for r in plaque_test if not r["gt"].any()]
    false_pos = [r for r in neg if r["pred"].any()]
    output["plaque"] = {
        "checkpoint_val_dice": plaque_state.get("val_dice"),
        "reproduced_val_train_resolution_letterbox_global_dice": summarize_seg(plaque_val)["dice"],
        "test": summarize_seg(plaque_test),
        "test_positive_n": len(pos), "test_negative_n": len(neg),
        "negative_images_with_any_false_positive": len(false_pos),
        "negative_image_false_positive_rate": len(false_pos) / len(neg) if neg else None,
        "threshold": 0.5, "min_component_area_px": 0,
        "test_status": output["split_checks"]["plaque_v3_test_status"],
    }
    for r in sorted(pos, key=lambda x: x["dice"])[:2]:
        write_overlay(r, f"plaque_failure_{r['path'].stem}_dice{r['dice']:.2f}.png")
    for r in sorted(pos, key=lambda x: x["dice"], reverse=True)[:2]:
        write_overlay(r, f"plaque_good_{r['path'].stem}_dice{r['dice']:.2f}.png")
    for r in false_pos[:2]:
        write_overlay(r, f"plaque_false_positive_{r['path'].stem}.png")
    del plaque_val, plaque_test, plaque_model
    gc.collect()

    # Wall checkpoint validation reproduction uses the training script default
    # (letterbox); held-out inference follows the current app path.
    wall_path = PROJECT / "models/wall_unet.pt"
    wall_state = torch.load(wall_path, map_location="cpu", weights_only=False)
    wall_model = load_model(str(wall_path), "cpu")
    cubs_data = DATA / "cubs"
    wall_val_letterbox = infer(wall_model, read_pairs(cubs_data, "val", "_wall.png"), True)
    wall_test_pairs = read_pairs(cubs_data, "test", "_wall.png")
    wall_test = infer(wall_model, wall_test_pairs)
    train_hashes = {hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (cubs_data / "train").glob("*.png") if "_wall" not in p.name}
    duplicate_test_names = {p.name for p, _, _ in wall_test_pairs
                            if hashlib.sha256(p.read_bytes()).hexdigest() in train_hashes}
    wall_test_clean = [r for r in wall_test if r["path"].name not in duplicate_test_names]
    manifest = json.loads((cubs_data / "manifest.json").read_text())
    errors, errors_clean, failures = [], [], 0
    for row in wall_test:
        meta = manifest[f"test/{row['path'].stem}"]
        measured = imt_from_wall_mask(row["pred"].astype(np.uint8) * 255,
                                      float(meta["cf_mm_px"]), prob=row["prob"])
        pred_mm = measured.get("mean_imt_mm")
        gt_mm = float(meta["gt_imt_mm"])
        row["gt_imt_mm"] = gt_mm
        row["pred_imt_mm"] = pred_mm
        if pred_mm is None or measured.get("measurement_failed"):
            failures += 1
        else:
            row["imt_abs_error_mm"] = abs(float(pred_mm) - gt_mm)
            errors.append(row["imt_abs_error_mm"])
            if row["path"].name not in duplicate_test_names:
                errors_clean.append(row["imt_abs_error_mm"])
    output["wall"] = {
        "checkpoint_val_dice": wall_state.get("val_dice"),
        "checkpoint_resize_mode": wall_state.get("resize_mode"),
        "training_default_resize_mode": "letterbox",
        "reproduced_val_train_resolution_letterbox_global_dice": summarize_seg(wall_val_letterbox)["dice"],
        "test": summarize_seg(wall_test),
        "exact_duplicate_test_images_excluded": sorted(duplicate_test_names),
        "test_after_excluding_exact_train_image_duplicates": summarize_seg(wall_test_clean),
        "test_imt_reference_n": len(errors), "test_imt_measurement_failures": failures,
        "test_imt_mae_mm": float(np.mean(errors)) if errors else None,
        "test_imt_mae_um": float(1000 * np.mean(errors)) if errors else None,
        "test_after_duplicate_exclusion_imt_reference_n": len(errors_clean),
        "test_after_duplicate_exclusion_imt_mae_mm": float(np.mean(errors_clean)) if errors_clean else None,
        "test_after_duplicate_exclusion_imt_mae_um": float(1000 * np.mean(errors_clean)) if errors_clean else None,
        "test_imt_metric": "app wall-mask geometry versus manifest gt_imt_mm using per-image cf_mm_px",
    }
    for p in OUT.glob("wall_*.png"):
        p.unlink()
    for r in sorted(wall_test_clean, key=lambda x: x["dice"])[:2]:
        write_overlay(r, f"wall_failure_{r['path'].stem}_dice{r['dice']:.2f}.png")
    for r in sorted(wall_test_clean, key=lambda x: x["dice"], reverse=True)[:2]:
        write_overlay(r, f"wall_good_{r['path'].stem}_dice{r['dice']:.2f}.png")

    (OUT / "metrics.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"Saved overlays to {OUT}")


if __name__ == "__main__":
    main()
