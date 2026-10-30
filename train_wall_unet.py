"""Train the vessel-wall (intima-media complex) segmentation U-Net on CUBS.

Dataset layout (from prepare_cubs.py):
    data/cubs/{train,val,test}/{N}.png + {N}_wall.png
    (600x800 grayscale, binary {0,255} IMC masks; manifest.json has CF + GT IMT)

Same U-Net family as the plaque model; supports --arch/--base,
fine-tuning (--init-from) and resume (--resume).

Usage:
    python train_wall_unet.py --data data/cubs --epochs 50 --size 320
    python train_wall_unet.py --resume --last models/last_wall.pt
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


class WallDataset(Dataset):
    """CUBS layout: <split>/{N}.png + <split>/{N}_wall.png."""

    def __init__(self, root: str, split: str, size: int = 320, augment: bool = False, resize_mode: str = "letterbox"):
        d = os.path.join(root, split)
        self.items = []
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".png") or "_wall" in f:
                continue
            stem = os.path.splitext(f)[0]
            m = os.path.join(d, f"{stem}_wall.png")
            if not os.path.exists(m):
                raise FileNotFoundError(f"Missing mask file: {m} for image: {os.path.join(d, f)}")
            self.items.append((os.path.join(d, f), m))
        self.size = size
        self.augment = augment
        self.resize_mode = resize_mode

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
        img = resize_input(img, self.size, self.resize_mode)
        mask = resize_input(mask, self.size, self.resize_mode, mask=True)
        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
            if random.random() < 0.5:
                ang = random.uniform(-8, 8)
                M = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), ang, 1.0)
                img = cv2.warpAffine(img, M, (self.size, self.size),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (self.size, self.size),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if random.random() < 0.5:
                img = np.clip(img * random.uniform(0.85, 1.15)
                              + random.uniform(-0.05, 0.05), 0, 1).astype(np.float32)
        return torch.from_numpy(img[None]).float(), torch.from_numpy(mask[None]).float()


@torch.no_grad()
def dice_score(model, loader, device) -> float:
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
    ap.add_argument("--data", default="data/cubs")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--arch", default="unet", choices=["unet", "attnunet", "unetpp"])
    ap.add_argument("--loss", default="bce_dice", choices=["bce_dice", "tversky", "focal_tversky"])
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--out", default="models/wall_unet.pt")
    ap.add_argument("--last", default="models/last_wall.pt")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--resume", action="store_true")
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

    train_ds = WallDataset(args.data, "train", args.size, augment=True, resize_mode=args.resize_mode)
    val_ds = WallDataset(args.data, "val", args.size, resize_mode=args.resize_mode)
    print(f"train={len(train_ds)} val={len(val_ds)} size={args.size}", flush=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers)

    if not len(train_ds) or not len(val_ds):
        raise ValueError("Empty train/val split")
    model = build_model(args.arch, args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"arch={args.arch} base={args.base} params={n_params:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")
    stale = 0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch, best = 1, -1.0
    if args.resume and os.path.exists(args.last):
        ckpt = torch.load(args.last, map_location=device)
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
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"fine-tuning from {args.init_from} "
              f"(val_dice={ckpt.get('val_dice', float('nan')):.4f})", flush=True)

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
        val_dice = dice_score(model, val_dl, device)
        sched.step()
        print(f"epoch {epoch:3d} loss={tot / n:.4f} val_dice={val_dice:.4f} "
              f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        if val_dice > best:
            best = val_dice
            stale = 0
            torch.save({"model": model.state_dict(), "val_dice": best,
                        "size": args.size, "arch": args.arch,
                        "base": args.base, "resize_mode": args.resize_mode, "seed": args.seed}, args.out)
        else:
            stale += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch,
                    "best": best, "val_dice": val_dice,
                    "size": args.size, "arch": args.arch,
                    "base": args.base, "resize_mode": args.resize_mode, "seed": args.seed}, args.last)
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
    print(f"best val Dice = {best:.4f} saved to {args.out}")


if __name__ == "__main__":
    main()
