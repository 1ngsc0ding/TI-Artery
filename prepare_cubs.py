"""Prepare CUBS for wall (intima-media complex) segmentation training.

Reads the CUBS technical dataset:
  images/tech_XXX.tiff                      (600x800 grayscale)
  LIMA-Profiles-interpolated/Manual-A1/tech_XXX-{LI,MA}.txt
      line1: x coords (int columns), line2: y coords (float)
  CF/tech_XXX_CF.txt                        (mm/pixel)
  Folds/f0/{Train,Val,Test}List.txt         (filtered to tech_*)

For each image, rasterizes the LI/MA polylines into an IMC region mask
(pixels between LI and MA boundaries), and writes:
  data/cubs/{train,val,test}/{N}.png          (image)
  data/cubs/{train,val,test}/{N}_wall.png     (IMC mask)
  data/cubs/manifest.json                     (CF + GT IMT per image)

Usage:
    .venv/bin/python prepare_cubs.py
"""
import json
import os

import cv2
import numpy as np

SRC = "data/cubs/m7ndn58sv6-1/DATASET_CUBS_tech"
OUT = "data/cubs"
FOLD = "f0"


def load_profile(path):
    a = np.loadtxt(path)
    return a[0].astype(int), a[1]  # xs, ys


def rasterize_imc(xs_li, ys_li, xs_ma, ys_ma, h, w):
    """Fill pixels between LI (top) and MA (bottom) boundaries."""
    mask = np.zeros((h, w), dtype=np.uint8)
    # both profiles share the same x grid in the interpolated version
    for x, y_top, y_bot in zip(xs_li, ys_li, ys_ma):
        if 0 <= x < w:
            t, b = int(round(min(y_top, y_bot))), int(round(max(y_top, y_bot)))
            t, b = max(t, 0), min(b, h - 1)
            if b >= t:
                mask[t:b + 1, x] = 255
    return mask


def main():
    # fold split, tech_ images only
    splits = {}
    for split, fname in [("train", "TrainList.txt"), ("val", "ValList.txt"),
                         ("test", "TestList.txt")]:
        ids = []
        with open(os.path.join(SRC, "Folds", FOLD, fname)) as f:
            for line in f:
                name = line.strip().replace(".png", "")
                if name.startswith("tech_"):
                    ids.append(name)
        splits[split] = sorted(ids)
        print(f"{split}: {len(ids)} images")

    manifest = {}
    for split, ids in splits.items():
        d = os.path.join(OUT, split)
        os.makedirs(d, exist_ok=True)
        for i, name in enumerate(ids):
            img = cv2.imread(os.path.join(SRC, "images", f"{name}.tiff"),
                             cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            xs_li, ys_li = load_profile(
                os.path.join(SRC, "LIMA-Profiles-interpolated",
                             "Manual-A1", f"{name}-LI.txt"))
            xs_ma, ys_ma = load_profile(
                os.path.join(SRC, "LIMA-Profiles-interpolated",
                             "Manual-A1", f"{name}-MA.txt"))
            mask = rasterize_imc(xs_li, ys_li, xs_ma, ys_ma, h, w)
            cf = float(open(os.path.join(SRC, "CF", f"{name}_CF.txt"))
                       .read().split()[0])
            # GT IMT: mean vertical distance LI->MA over annotated columns
            gt_imt_px = float(np.mean(np.abs(ys_ma - ys_li)))
            cv2.imwrite(os.path.join(d, f"{i:03d}.png"), img)
            cv2.imwrite(os.path.join(d, f"{i:03d}_wall.png"), mask)
            manifest[f"{split}/{i:03d}"] = {
                "src": name, "cf_mm_px": cf,
                "gt_imt_mm": gt_imt_px * cf,
                "wall_frac": float((mask > 0).mean()),
            }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    fracs = [m["wall_frac"] for m in manifest.values()]
    imts = [m["gt_imt_mm"] for m in manifest.values()]
    print(f"done. wall covers {np.mean(fracs)*100:.1f}% of pixels "
          f"(range {np.min(fracs)*100:.1f}-{np.max(fracs)*100:.1f}%)")
    print(f"GT IMT: mean {np.mean(imts):.3f} mm "
          f"(range {np.min(imts):.3f}-{np.max(imts):.3f})")


if __name__ == "__main__":
    main()
