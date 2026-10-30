"""Step 7 of the TI-Artery pipeline: plaque segmentation with U-Net.

The AI's job is ONLY to draw boundaries (segmentation). Area/length are
computed geometrically from the mask (planimetry principle) in plaque.py.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from .geometry import resize_input, restore_probability


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Standard encoder-decoder U-Net with skip connections."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        chs = [base * (2 ** i) for i in range(4)]  # 32,64,128,256
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.downs.append(DoubleConv(prev, c))
            self.pools.append(nn.MaxPool2d(2))
            prev = c
        self.bottleneck = DoubleConv(prev, prev * 2)
        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        prev_up = prev * 2
        for c in reversed(chs):
            self.upconvs.append(nn.ConvTranspose2d(prev_up, c, 2, stride=2))
            self.ups.append(DoubleConv(c * 2, c))
            prev_up = c
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, up, skip in zip(self.upconvs, self.ups, reversed(skips)):
            x = upconv(x)
            if x.shape[2:] != skip.shape[2:]:  # guard against odd sizes
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = up(torch.cat([x, skip], dim=1))
        return self.head(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., Attention U-Net).

    Learns to suppress irrelevant background (e.g. bright vessel wall)
    in the skip connection, letting the decoder focus on plaque.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = self.psi(self.relu(g1 + x1))
        return x * alpha


class AttentionUNet(nn.Module):
    """U-Net with attention gates on every skip connection.

    Shared module names (downs/pools/bottleneck/upconvs/ups/head) match
    plain UNet so weights can be warm-started from a UNet checkpoint
    with strict=False; the attention gates train from scratch.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        chs = [base * (2 ** i) for i in range(4)]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.downs.append(DoubleConv(prev, c))
            self.pools.append(nn.MaxPool2d(2))
            prev = c
        self.bottleneck = DoubleConv(prev, prev * 2)
        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.attns = nn.ModuleList()
        prev_up = prev * 2
        for c in reversed(chs):
            self.upconvs.append(nn.ConvTranspose2d(prev_up, c, 2, stride=2))
            self.attns.append(AttentionGate(F_g=c, F_l=c, F_int=max(c // 2, 8)))
            self.ups.append(DoubleConv(c * 2, c))
            prev_up = c
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, attn, up, skip in zip(
                self.upconvs, self.attns, self.ups, reversed(skips)):
            g = upconv(x)
            if g.shape[2:] != skip.shape[2:]:
                g = nn.functional.interpolate(g, size=skip.shape[2:])
            skip_att = attn(g, skip)
            x = up(torch.cat([g, skip_att], dim=1))
        return self.head(x)


class NestedUNet(nn.Module):
    """Compact U-Net++ (nested skip pathways). PDF allows U-Net variants."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32, deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        nb = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv0_0 = DoubleConv(in_ch, nb[0])
        self.conv1_0 = DoubleConv(nb[0], nb[1])
        self.conv2_0 = DoubleConv(nb[1], nb[2])
        self.conv3_0 = DoubleConv(nb[2], nb[3])
        self.conv4_0 = DoubleConv(nb[3], nb[4])
        self.conv0_1 = DoubleConv(nb[0] + nb[1], nb[0])
        self.conv1_1 = DoubleConv(nb[1] + nb[2], nb[1])
        self.conv2_1 = DoubleConv(nb[2] + nb[3], nb[2])
        self.conv3_1 = DoubleConv(nb[3] + nb[4], nb[3])
        self.conv0_2 = DoubleConv(nb[0] * 2 + nb[1], nb[0])
        self.conv1_2 = DoubleConv(nb[1] * 2 + nb[2], nb[1])
        self.conv2_2 = DoubleConv(nb[2] * 2 + nb[3], nb[2])
        self.conv0_3 = DoubleConv(nb[0] * 3 + nb[1], nb[0])
        self.conv1_3 = DoubleConv(nb[1] * 3 + nb[2], nb[1])
        self.conv0_4 = DoubleConv(nb[0] * 4 + nb[1], nb[0])
        self.head = nn.Conv2d(nb[0], out_ch, 1)
        self.heads = nn.ModuleList([nn.Conv2d(nb[0], out_ch, 1) for _ in range(4)]) if deep_supervision else None

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        if self.deep_supervision and self.training:
            return [h(x) for h, x in zip(self.heads, [x0_1, x0_2, x0_3, x0_4])]
        return self.head(x0_4)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def bce_dice_loss(logits, target, bce_w: float = 0.5, pos_weight: float = 1.0):
    pw = torch.tensor(pos_weight, device=logits.device)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    return bce_w * bce + (1 - bce_w) * dice_loss(logits, target)


def tversky_loss(logits: torch.Tensor, target: torch.Tensor,
                 alpha: float = 0.4, beta: float = 0.6, eps: float = 1e-6) -> torch.Tensor:
    """Tversky: alpha weights FP, beta weights FN. beta>alpha favors recall."""
    prob = torch.sigmoid(logits)
    dims = (2, 3)
    tp = (prob * target).sum(dim=dims)
    fp = (prob * (1 - target)).sum(dim=dims)
    fn = ((1 - prob) * target).sum(dim=dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - tversky.mean()


def focal_tversky_loss(logits, target, alpha=0.4, beta=0.6, gamma: float = 0.75):
    t = tversky_loss(logits, target, alpha=alpha, beta=beta)
    return torch.pow(t, gamma)


def segmentation_loss(logits, target, kind: str = "bce_dice", pos_weight: float = 1.0):
    if isinstance(logits, list):
        return sum(segmentation_loss(x, target, kind, pos_weight) for x in logits) / len(logits)
    if kind == "tversky":
        return 0.3 * bce_dice_loss(logits, target, pos_weight=pos_weight) + 0.7 * tversky_loss(logits, target)
    if kind == "focal_tversky":
        return 0.25 * bce_dice_loss(logits, target, pos_weight=pos_weight) + 0.75 * focal_tversky_loss(logits, target)
    return bce_dice_loss(logits, target, pos_weight=pos_weight)


@torch.no_grad()
def predict_prob(model: nn.Module, gray: np.ndarray,
                 size: int = 256, device: str = "cpu",
                 tta: bool = False) -> np.ndarray:
    """Probability map at original resolution. Inference is on raw pixels."""
    size = getattr(model, "input_size", size)
    mode = getattr(model, "resize_mode", "stretch")
    img = np.clip(gray, 0, 1)
    resized = resize_input(img, size, mode)
    x = torch.from_numpy(resized[None, None]).float().to(device)
    model.eval()
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    if tta:
        xf = torch.flip(x, dims=[-1])
        pf = torch.flip(torch.sigmoid(model(xf))[0, 0], dims=[-1]).cpu().numpy()
        prob = 0.5 * (prob + pf)
    return restore_probability(prob, gray.shape, mode)


@torch.no_grad()
def predict_mask(model: nn.Module, gray: np.ndarray,
                 size: int = 256, thr: float = 0.5,
                 min_area: int = 0,
                 device: str = "cpu",
                 tta: bool = False,
                 return_prob: bool = False):
    """Run the model on a float grayscale image; return uint8 mask at original size.

    NOTE: no Lee filtering here - the model was trained on raw resized
    images, so inference must match training preprocessing exactly.
    min_area: drop predicted components smaller than this many px
    (speckle FP cleanup; real plaques are far larger).
    """
    size = getattr(model, "input_size", size)
    prob = predict_prob(model, gray, size=size, device=device, tta=tta)
    mask = (prob >= thr).astype(np.uint8) * 255
    if min_area > 0:
        num, lab, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        keep = np.zeros_like(mask, dtype=bool)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[lab == i] = True
        mask = keep.astype(np.uint8) * 255
    if return_prob:
        return mask, prob
    return mask


def build_model(arch: str = "unet", base: int = 32,
               in_ch: int = 1, out_ch: int = 1) -> nn.Module:
    """Build a segmentation model by architecture name."""
    if arch == "attnunet":
        return AttentionUNet(in_ch=in_ch, out_ch=out_ch, base=base)
    if arch in {"unetpp", "unet++"}:
        return NestedUNet(in_ch=in_ch, out_ch=out_ch, base=base)
    return UNet(in_ch=in_ch, out_ch=out_ch, base=base)


def load_model(path: str, device: str = "cpu", arch: str = "unet",
               base: int = 32) -> nn.Module:
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"Unexpected checkpoint format: {path}")
    model = build_model(state.get("arch", arch), state.get("base", base))
    model.input_size = state.get("size", 320)
    model.resize_mode = state.get("resize_mode", "stretch")
    model.threshold = float(state.get("threshold", 0.5))
    model.load_state_dict(state["model"])
    return model.to(device).eval()
