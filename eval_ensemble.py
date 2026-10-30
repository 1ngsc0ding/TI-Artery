"""Ensemble eval: average probability maps of multiple models, then
threshold + min-area cleanup. Quick to run (inference only).

Usage:
    .venv/bin/python eval_ensemble.py --out eval_ensemble
"""
import argparse
import os

import cv2
import numpy as np
import torch

from tiartery.segmentation import build_model
from eval_plaque import dice, remove_small_components, predict_prob


MODELS = [
    ("models/plaque_unet.pt", "unet", 32),
    ("models/plaque_unet48.pt", "unet", 48),
    ("models/last_attn.pt", "attnunet", 32),  # attn never beat baseline; use last
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ar-plaquesegm1/test")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--min-area", type=int, default=1500)
    ap.add_argument("--out", default="eval_ensemble")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for path, arch, base in MODELS:
        if not os.path.exists(path):
            print(f"skip missing {path}")
            continue
        ckpt = torch.load(path, map_location=device)
        m = build_model(arch, base).to(device).eval()
        m.load_state_dict(ckpt["model"], strict=False)
        models.append((path, m))
        print(f"loaded {path} (val={ckpt.get('val_dice', float('nan')):.4f})")

    pairs = []
    for f in sorted(os.listdir(args.data)):
        if f.endswith(".png") and "_labeled" not in f:
            stem = os.path.splitext(f)[0].strip()
            m = os.path.join(args.data, f"{stem}_labeled.png")
            if os.path.exists(m):
                pairs.append((os.path.join(args.data, f), m))

    cache = []
    with torch.no_grad():
        for ip, mp in pairs:
            img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127
            probs = [predict_prob(m, img, args.size, device) for _, m in models]
            cache.append((ip, img, gt, np.mean(probs, axis=0)))

    for thr in [0.4, 0.5, 0.6, 0.7]:
        ds = [dice(remove_small_components(p >= thr, args.min_area), g)
              for _, _, g, p in cache]
        pos = [d for (_, _, g, _), d in zip(cache, ds) if g.sum() > 0]
        neg = [d for (_, _, g, _), d in zip(cache, ds) if g.sum() == 0]
        print(f"thr={thr}: overall={np.mean(ds):.4f} "
              f"positive={np.mean(pos):.4f} empty={np.mean(neg):.4f}")


if __name__ == "__main__":
    main()
