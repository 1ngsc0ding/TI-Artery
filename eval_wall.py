from tiartery.geometry import resize_input, restore_probability
"""Evaluate the wall (IMC) segmentation model.

Metrics:
  - Dice on the intima-media complex region
  - IMT error: mean absolute error between predicted and GT IMT (mm).
    IMT is derived from the mask as the mean vertical wall thickness
    over annotated columns x pixel-to-mm conversion factor.

Usage:
    .venv/bin/python eval_wall.py --model models/wall_unet.pt --out eval_wall
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from tiartery.segmentation import build_model


def dice(p, g, eps=1e-6):
    p, g = p.astype(bool), g.astype(bool)
    inter = np.logical_and(p, g).sum()
    return float(2 * inter / (p.sum() + g.sum() + eps))


def imt_from_mask(mask: np.ndarray, cf: float) -> float:
    """Mean vertical wall thickness (px) x mm/px. NaN if no wall predicted."""
    cols = np.maximum(mask.sum(axis=0).astype(float) - 1, 0)
    cols = cols[cols > 0]
    if len(cols) == 0:
        return float("nan")
    return float(cols.mean() * cf)


@torch.no_grad()
def predict_prob(model, img, size, device):
    resized = resize_input(img, size, getattr(model, "resize_mode", "stretch"))
    x = torch.from_numpy(resized[None, None]).float().to(device)
    prob_small = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    h, w = img.shape
    return restore_probability(prob_small, (h, w), getattr(model, "resize_mode", "stretch"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/wall_unet.pt")
    ap.add_argument("--arch", default="unet", choices=["unet", "attnunet"])
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--data", default="data/cubs/val")
    ap.add_argument("--allow-test", action="store_true",
                    help="Required to evaluate the held-out test folder")
    ap.add_argument("--manifest", default="data/cubs/manifest.json")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--out", default="eval_wall")
    args = ap.parse_args()

    if os.path.basename(os.path.normpath(args.data)) == "test" and not args.allow_test:
        ap.error("Refusing held-out test without --allow-test (use val for tuning)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device)
    model = build_model(ckpt.get("arch", args.arch), ckpt.get("base", args.base)).to(device).eval()
    model.resize_mode = ckpt.get("resize_mode", "stretch")
    args.size = ckpt.get("size", args.size)
    model.load_state_dict(ckpt["model"])
    print(f"model {args.model} (ckpt val_dice={ckpt.get('val_dice', float('nan')):.4f})")

    manifest = json.load(open(args.manifest))
    split = os.path.basename(os.path.normpath(args.data))
    pairs = []
    for f in sorted(os.listdir(args.data)):
        if f.endswith(".png") and "_wall" not in f:
            stem = os.path.splitext(f)[0]
            m = os.path.join(args.data, f"{stem}_wall.png")
            if os.path.exists(m):
                key = f"{split}/{stem}"
                pairs.append((os.path.join(args.data, f), m,
                              manifest[key]["cf_mm_px"],
                              manifest[key]["gt_imt_mm"]))

    print(f"test pairs: {len(pairs)}")
    dices, errs = [], []
    for ip, mp, cf, gt_imt in pairs:
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127
        prob = predict_prob(model, img, args.size, device)
        # threshold sweep-free: 0.5 is fine for region segmentation
        pred = prob >= 0.5
        dices.append(dice(pred, gt))
        pred_imt = imt_from_mask(pred, cf)
        if not np.isnan(pred_imt):
            errs.append(abs(pred_imt - gt_imt))

    print(f"IMT failures = {len(pairs)-len(errs)}/{len(pairs)}")
    errs = np.array(errs)
    print(f"\nWall segmentation (native 600x800, infer {args.size}px):")
    print(f"  mean Dice        = {np.mean(dices):.4f}  (n={len(dices)})")
    print(f"  IMT MAE          = {np.mean(errs)*1000:.0f} um  (n={len(errs)})")
    print(f"  IMT MAE < 150 um = {(errs < 0.150).mean()*100:.0f}% of images")
    print("  (reference: CUBS inter-analyst variability ~160 um)")

    # overlays: 3 worst + 3 best by Dice
    os.makedirs(args.out, exist_ok=True)
    scored = sorted(zip(dices, pairs), key=lambda s: s[0])
    for d, (ip, mp, cf, gt_imt) in scored[:3] + scored[-3:]:
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127
        prob = predict_prob(model, img, args.size, device)
        pred = prob >= 0.5
        rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        rgb[gt] = (0, 255, 0)
        rgb[np.logical_and(pred, ~gt)] = (0, 0, 255)
        rgb[np.logical_and(~pred, gt)] = (255, 0, 0)
        name = os.path.splitext(os.path.basename(ip))[0]
        cv2.imwrite(os.path.join(args.out, f"{name}_dice{d:.2f}.png"), rgb)
    print(f"overlays -> {args.out}/")


if __name__ == "__main__":
    main()
