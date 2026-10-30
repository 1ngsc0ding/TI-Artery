from tiartery.geometry import resize_input, restore_probability
"""Evaluate the plaque U-Net the way the pipeline uses it.

- Inference at 320px (training resolution), mask resized back to native 800px.
- Fixed Dice: empty GT + empty prediction = 1.0 (was wrongly 0.0 before).
- Threshold sweep, per-image Dice/F1, plaque-positive vs empty split,
  precision/recall, overlay figures.

Usage:
    .venv/bin/python eval_plaque.py --model models/plaque_unet.pt --out eval_out
"""
import argparse
import os

import cv2
import numpy as np
import torch

from tiartery.segmentation import UNet  # noqa: F401  (kept for compat)


def dice(p, g, eps=1e-6):
    p, g = p.astype(bool), g.astype(bool)
    if g.sum() == 0:
        return 1.0 if p.sum() == 0 else 0.0
    inter = np.logical_and(p, g).sum()
    return float(2 * inter / (p.sum() + g.sum() + eps))


@torch.no_grad()
def predict_prob(model, img, size, device):
    resized = resize_input(img, size, getattr(model, "resize_mode", "stretch"))
    x = torch.from_numpy(resized[None, None]).float().to(device)
    prob_small = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    h, w = img.shape
    return restore_probability(prob_small, (h, w), getattr(model, "resize_mode", "stretch"))


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than min_area (speckle FP cleanup)."""
    if min_area <= 0:
        return mask
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/plaque_unet.pt")
    ap.add_argument("--arch", default="unet", choices=["unet", "attnunet", "unetpp"])
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--data", default="data/ar-plaquesegm1/test")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--min-area", type=int, default=0,
                    help="drop predicted components smaller than this (px at 800px)")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tune", action="store_true", help="Validation folder only")
    ap.add_argument("--allow-test", action="store_true",
                    help="Required to evaluate a held-out test folder")
    args = ap.parse_args()
    split_name = os.path.basename(os.path.normpath(args.data))
    if split_name == "test" and not args.allow_test:
        ap.error("Refusing held-out test without --allow-test (tune on val)")
    if args.tune and os.path.basename(os.path.normpath(args.data)) != "val":
        ap.error("Threshold tuning is only allowed on a val folder")
    if not 0 <= args.threshold <= 1:
        ap.error("threshold must be between 0 and 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device)
    from tiartery.segmentation import build_model
    model = build_model(ckpt.get("arch", args.arch), ckpt.get("base", args.base)).to(device).eval()
    model.resize_mode = ckpt.get("resize_mode", "stretch")
    args.size = ckpt.get("size", args.size)
    model.load_state_dict(ckpt["model"])
    print(f"model {args.model} (ckpt val_dice={ckpt.get('val_dice', float('nan')):.4f})")

    pairs = []
    for f in sorted(os.listdir(args.data)):
        if f.endswith(".png") and "_labeled" not in f:
            stem = os.path.splitext(f)[0].strip()
            m = os.path.join(args.data, f"{stem}_labeled.png")
            if os.path.exists(m):
                pairs.append((os.path.join(args.data, f), m))
    print(f"test pairs: {len(pairs)}")

    # cache probabilities once
    cache = []
    for ip, mp in pairs:
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127
        prob = predict_prob(model, img, args.size, device)
        cache.append((ip, img, gt, prob))

    if not cache:
        raise ValueError("No evaluation pairs found")
    thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7] if args.tune else [args.threshold]
    areas = [0, 80, 200, 400, 800, 1500] if args.tune else [args.min_area]
    sweep = {}
    for thr in thresholds:
        for area in areas:
            tp = fp = fn = 0
            for _, _, gt, prob in cache:
                p = remove_small_components(prob >= thr, area)
                tp += np.logical_and(p, gt).sum()
                fp += np.logical_and(p, ~gt).sum()
                fn += np.logical_and(~p, gt).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            sweep[(thr, area)] = float(f1)
            print(f"  thr={thr} min_area={area}: global pixel F1 = {f1:.4f}")
    best_thr, best_area = max(sweep, key=sweep.get)
    args.min_area = best_area

    masks = [remove_small_components(prob >= best_thr, args.min_area)
             for _, _, _, prob in cache]
    dices = [dice(m, gt) for (_, _, gt, _), m in zip(cache, masks)]
    pos = [d for (_, _, gt, _), d in zip(cache, dices) if gt.sum() > 0]
    neg = [d for (_, _, gt, _), d in zip(cache, dices) if gt.sum() == 0]
    precs, recs = [], []
    for (_, _, gt, _), p in zip(cache, masks):
        tp = np.logical_and(p, gt).sum()
        precs.append(tp / max(p.sum(), 1))
        recs.append(tp / max(gt.sum(), 1))
    print(f"\nFINAL (native 800px, infer {args.size}px, thr={best_thr}, "
          f"min_area={args.min_area}):")
    print(f"  mean Dice/F1      = {np.mean(dices):.4f}  (n={len(dices)})")
    print(f"  plaque-positive   = {np.mean(pos):.4f}  (n={len(pos)})")
    print(f"  empty (no plaque) = {np.mean(neg):.4f}  (n={len(neg)})")
    print(f"  mean precision = {np.mean(precs):.4f}, mean recall = {np.mean(recs):.4f}")

    tp = sum(np.logical_and(m, gt).sum() for (_, _, gt, _), m in zip(cache, masks))
    fp = sum(np.logical_and(m, ~gt).sum() for (_, _, gt, _), m in zip(cache, masks))
    fn = sum(np.logical_and(~m, gt).sum() for (_, _, gt, _), m in zip(cache, masks))
    print(f"  global pixel F1 = {2*tp/max(2*tp+fp+fn, 1):.4f}")
    actual = np.array([bool(gt.any()) for _, _, gt, _ in cache])
    detected = np.array([bool(m.any()) for m in masks])
    dtp, dfp, dfn = (detected & actual).sum(), (detected & ~actual).sum(), (~detected & actual).sum()
    print(f"  image detection F1 = {2*dtp/max(2*dtp+dfp+dfn, 1):.4f}")
    print(f"  image detection precision={dtp/max(dtp+dfp,1):.4f} recall={dtp/max(dtp+dfn,1):.4f}")

    # overlays: 3 worst + 3 best plaque-positive
    os.makedirs(args.out, exist_ok=True)
    scored = sorted(
        [(d, ip, img, gt, m)
         for (ip, img, gt, _), d, m in zip(cache, dices, masks) if gt.sum() > 0],
        key=lambda s: s[0])
    for d, ip, img, gt, p in scored[:3] + scored[-3:]:
        rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        rgb[gt] = (0, 255, 0)
        rgb[np.logical_and(p, ~gt)] = (0, 0, 255)
        rgb[np.logical_and(~p, gt)] = (255, 0, 0)
        name = os.path.splitext(os.path.basename(ip))[0].strip()
        cv2.imwrite(os.path.join(args.out, f"{name}_dice{d:.2f}.png"), rgb)
    print(f"overlays -> {args.out}/")


if __name__ == "__main__":
    main()
