"""Train the plaque-segmentation U-Net on Ar-PlaqSegm1.

Dataset layout: data/ar-plaquesegm1/{train,test}/{N}.png + {N}_labeled.png
(flat, 800x800 grayscale PNGs, binary {0,255} masks).

Supports fine-tuning from a checkpoint (--init-from) and resuming an
interrupted run (--resume, via models/plaque_unet_last.pt saved each epoch).

Usage:
    python train_plaque_unet.py --data data/ar-plaquesegm1 --epochs 50 --size 320
    python train_plaque_unet.py --resume   # continue after interruption
"""
import argparse
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tiartery.segmentation import build_model, segmentation_loss
from tiartery.geometry import resize_input

LAST_PATH = "models/plaque_unet_last.pt"


class PlaqueDataset(Dataset):
    """Flat layout: <split>/{N}.png + <split>/{N}_labeled.png (Ar-PlaqSegm1).

    Handles quirks: RGB masks (IMREAD_GRAYSCALE normalizes), filenames
    with stray whitespace (e.g. '544 .png'), empty masks (no plaque).
    """

    def __init__(self, root: str, split: str, size: int = 320, augment: bool = False,
                 resize_mode: str = "letterbox", copy_paste: bool = False):
        d = os.path.join(root, split)
        self.items = []
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".png") or "_labeled" in f:
                continue
            stem = os.path.splitext(f)[0].strip()
            m = os.path.join(d, f"{stem}_labeled.png")
            if not os.path.exists(m):
                raise FileNotFoundError(f"Missing mask file: {m} for image: {os.path.join(d, f)}")
            self.items.append((os.path.join(d, f), m))
        self.size = size
        self.augment = augment
        self.resize_mode = resize_mode
        self.copy_paste = copy_paste
        self.pos_idx = []
        if copy_paste:
            for i, (_, mp) in enumerate(self.items):
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m is not None and (m > 127).sum() > 80:
                    self.pos_idx.append(i)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise OSError(f"Cannot read image file: {ip}")
        if mask is None:
            raise OSError(f"Cannot read mask file: {mp}")
        if img.shape != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: image={ip}, mask={mp}")
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        if self.augment and self.copy_paste and self.pos_idx and random.random() < 0.45:
            img, mask = self._copy_paste(img, mask)
        img = resize_input(img, self.size, self.resize_mode)
        mask = resize_input(mask, self.size, self.resize_mode, mask=True)
        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
            if random.random() < 0.35:
                img, mask = img[::-1].copy(), mask[::-1].copy()
            if random.random() < 0.7:
                ang = random.uniform(-18, 18)
                scale = random.uniform(0.85, 1.18)
                M = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), ang, scale)
                tx, ty = random.uniform(-0.08, 0.08) * self.size, random.uniform(-0.08, 0.08) * self.size
                M[0, 2] += tx
                M[1, 2] += ty
                img = cv2.warpAffine(img, M, (self.size, self.size),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (self.size, self.size),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if random.random() < 0.5:
                img = np.clip(img * random.uniform(0.75, 1.25)
                              + random.uniform(-0.08, 0.08), 0, 1).astype(np.float32)
            if random.random() < 0.35:
                gamma = random.uniform(0.7, 1.4)
                img = np.clip(img ** gamma, 0, 1).astype(np.float32)
            if random.random() < 0.3:
                u8 = (img * 255).astype(np.uint8)
                img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8).astype(np.float32) / 255.0
            if random.random() < 0.4:
                img = np.clip(img + np.random.normal(0, 0.03, img.shape).astype(np.float32), 0, 1)
        x = torch.from_numpy(img[None]).float()
        y = torch.from_numpy(mask[None]).float()
        return x, y

    def _copy_paste(self, img, mask):
        j = random.choice(self.pos_idx)
        src_img = cv2.imread(self.items[j][0], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        src_mask = (cv2.imread(self.items[j][1], cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        if src_img.shape != img.shape:
            src_img = cv2.resize(src_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            src_mask = cv2.resize(src_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(src_mask > 0.5)
        if len(xs) < 40:
            return img, mask
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        patch_i, patch_m = src_img[y0:y1, x0:x1], src_mask[y0:y1, x0:x1]
        h, w = patch_m.shape
        if h >= img.shape[0] or w >= img.shape[1]:
            return img, mask
        yy = random.randint(0, img.shape[0] - h)
        xx = random.randint(0, img.shape[1] - w)
        sel = patch_m > 0.5
        dst = img.copy()
        mdst = mask.copy()
        roi_i = dst[yy:yy + h, xx:xx + w]
        roi_m = mdst[yy:yy + h, xx:xx + w]
        roi_i[sel] = patch_i[sel]
        roi_m[sel] = 1.0
        dst[yy:yy + h, xx:xx + w] = roi_i
        mdst[yy:yy + h, xx:xx + w] = roi_m
        return dst, mdst


@torch.no_grad()
def dice_score(model, loader, device) -> float:
    """Mean Dice (= F1 for binary segmentation) over the loader."""
    model.eval()
    inter, union, eps = 0.0, 0.0, 1e-6
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        inter += (p * y).sum().item()
        union += p.sum().item() + y.sum().item()
    return (2 * inter + eps) / (union + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ar-plaquesegm1")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--out", default="models/plaque_unet.pt")
    ap.add_argument("--arch", default="unet", choices=["unet", "attnunet", "unetpp"],
                    help="model architecture")
    ap.add_argument("--loss", default="bce_dice",
                    choices=["bce_dice", "tversky", "focal_tversky"])
    ap.add_argument("--copy-paste", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop epochs with no val improvement (0=off)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=32,
                    help="base channels (model width)")
    ap.add_argument("--last", default=LAST_PATH,
                    help="per-epoch checkpoint path (for parallel/sequential runs)")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to fine-tune from")
    ap.add_argument("--no-carry-best", action="store_true",
                    help="don't carry the init checkpoint's best score "
                         "(track this run's own best)")
    ap.add_argument("--resume", action="store_true",
                    help="resume interrupted run from models/plaque_unet_last.pt")
    ap.add_argument("--resize-mode", choices=["letterbox", "stretch"], default="letterbox")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-f1", type=float, default=None)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    if args.target_f1 is not None and not 0 < args.target_f1 <= 1:
        ap.error("target-f1 must be in (0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.size < 32 or args.size % 16:
        ap.error("--size must be a positive multiple of 16, at least 32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    train_ds = PlaqueDataset(args.data, "train", args.size, augment=True,
                             resize_mode=args.resize_mode, copy_paste=args.copy_paste)
    test_ds = PlaqueDataset(args.data, "val", args.size, resize_mode=args.resize_mode)
    print(f"train={len(train_ds)} val={len(test_ds)} size={args.size}", flush=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    test_dl = DataLoader(test_ds, batch_size=args.batch, num_workers=args.workers)

    if not len(train_ds) or not len(test_ds):
        raise ValueError("Empty train/val split")
    model = build_model(args.arch, args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"arch={args.arch} base={args.base} params={n_params:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")
    stale = 0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    last_path = args.last
    start_epoch, best = 1, -1.0
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        if ckpt.get("resize_mode", "stretch") != args.resize_mode or ckpt.get("size", args.size) != args.size:
            raise ValueError("Resume preprocessing differs; use --init-from for a new experiment")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt.get("best", 0.0)
        print(f"resumed from epoch {ckpt['epoch']} (best={best:.4f})", flush=True)
    elif args.init_from and os.path.exists(args.init_from):
        ckpt = torch.load(args.init_from, map_location=device)
        # strict=False: AttentionUNet reuses UNet weights, gates train fresh
        model.load_state_dict(ckpt["model"], strict=False)
        if not args.no_carry_best:
            best = -1.0
        print(f"fine-tuning from {args.init_from} (val_dice={ckpt.get('val_dice', float('nan')):.4f})",
              flush=True)

    for checkpoint_path in [args.out, args.last]:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device == "cuda"):
                loss = segmentation_loss(model(x), y, kind=args.loss, pos_weight=args.pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
            n += len(x)
        val_dice = dice_score(model, test_dl, device)
        sched.step()
        print(f"epoch {epoch:3d} loss={tot / n:.4f} val_dice={val_dice:.4f} "
              f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        if val_dice > best:
            best = val_dice
            stale = 0
            torch.save({"model": model.state_dict(), "val_dice": best,
                        "size": args.size, "arch": args.arch,
                        "base": args.base, "resize_mode": args.resize_mode, "seed": args.seed,
                        "loss": args.loss, "threshold": 0.5}, args.out)
        else:
            stale += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "val_dice": val_dice,
                    "size": args.size, "arch": args.arch,
                    "base": args.base, "resize_mode": args.resize_mode, "seed": args.seed}, last_path)
        import json
        print("METRIC_JSON " + json.dumps({"epoch": epoch, "val_f1": val_dice,
              "best_val_f1": best, "checkpoint": os.path.abspath(args.out),
              "last_checkpoint": os.path.abspath(args.last),
              "metric": "validation_global_pixel_f1_training_resolution", "size": args.size}), flush=True)
        if args.target_f1 is not None and val_dice >= args.target_f1:
            print("Target validation F1 reached; checkpoints saved.", flush=True)
            break
        if args.patience > 0 and stale >= args.patience:
            print(f"Early stop at epoch {epoch}; best={best:.4f}", flush=True)
            break
    print(f"best val Dice (F1) = {best:.4f} saved to {args.out}")


if __name__ == "__main__":
    main()
