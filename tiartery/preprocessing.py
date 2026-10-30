"""Steps 2-3 of the TI-Artery pipeline: frame selection (quality control)
and speckle noise reduction (edge-preserving Lee filter).

Speckle in ultrasound is multiplicative noise, so generic Gaussian
smoothing blurs the wall/plaque boundaries we need. The Lee filter adapts
the smoothing strength from local statistics: flat regions are smoothed
aggressively, edge regions are preserved.
"""
import os

import cv2
import numpy as np


def load_grayscale(path: str) -> np.ndarray:
    """Load an image as float32 grayscale in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img.astype(np.float32) / 255.0


def frame_quality(gray: np.ndarray) -> dict:
    """Quality metrics for one B-mode frame (float image in [0, 1]).

    blur: variance of Laplacian (higher = sharper).
    brightness/contrast: mean / std of intensities.
    """
    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    blur = float(cv2.Laplacian(u8, cv2.CV_64F).var())
    return {
        "blur": blur,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
    }


def quality_issues(m: dict, blur_min: float = 30.0,
                   bright_lo: float = 0.05, bright_hi: float = 0.95,
                   contrast_min: float = 0.03) -> list[str]:
    """Plain-English reasons a frame failed QC (empty list = usable)."""
    issues = []
    blur = float(m["blur"])
    brightness = float(m["brightness"])
    contrast = float(m["contrast"])
    if blur < blur_min:
        issues.append(
            f"The image is too blurry for reliable outlines "
            f"(sharpness {blur:.1f}; need at least {blur_min:.0f}).")
    if brightness < bright_lo:
        issues.append(
            f"The image is too dark (brightness {brightness:.2f}; "
            f"need at least {bright_lo:.2f}).")
    elif brightness > bright_hi:
        issues.append(
            f"The image is too bright or washed out "
            f"(brightness {brightness:.2f}; need at most {bright_hi:.2f}).")
    if contrast < contrast_min:
        issues.append(
            f"The image has too little contrast "
            f"({contrast:.2f}; need at least {contrast_min:.2f}).")
    return issues


def is_usable(m: dict, blur_min: float = 30.0,
              bright_lo: float = 0.05, bright_hi: float = 0.95,
              contrast_min: float = 0.03) -> bool:
    """Reject blurred, saturated, or flat frames before analysis."""
    return not quality_issues(m, blur_min, bright_lo, bright_hi, contrast_min)


def select_best(frames: list, top_k: int = 1) -> list:
    """Return the top_k sharpest frames (for video input)."""
    ranked = sorted(frames, key=lambda f: frame_quality(f)["blur"], reverse=True)
    return ranked[:top_k]


def load_video_frames(path: str, max_frames: int = 60) -> list:
    """Sample grayscale frames from a B-mode video (AVI/MP4)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot read video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max_frames) if total > max_frames else 1
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame.astype(np.float32) / 255.0)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames


def load_ultrasound(path: str) -> tuple[np.ndarray, dict]:
    """Load a still image or the sharpest video frame. Returns (gray, meta)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}:
        frames = load_video_frames(path)
        best = select_best(frames, top_k=1)[0]
        return best, {"source": "video", "n_frames": len(frames)}
    return load_grayscale(path), {"source": "image", "n_frames": 1}


def srad(img: np.ndarray, niter: int = 15, lam: float = 0.15,
         delta: float = 2.0) -> np.ndarray:
    """Speckle Reducing Anisotropic Diffusion (Yu & Acton).

    Multiplicative-noise diffusion that smooths speckle while keeping
    lumen–intima and media–adventitia edges. Used for QC/display; plaque
    U-Net inference stays on raw pixels to match training.
    """
    img = np.clip(np.asarray(img, dtype=np.float64), 1e-6, None)
    rows, cols = img.shape
    for _ in range(max(1, niter)):
        padded = np.pad(img, 1, mode="edge")
        i_n = padded[:-2, 1:-1] - img
        i_s = padded[2:, 1:-1] - img
        i_e = padded[1:-1, 2:] - img
        i_w = padded[1:-1, :-2] - img
        q0 = 1.0 / max(delta, 1e-6)
        gradient2 = i_n ** 2 + i_s ** 2 + i_e ** 2 + i_w ** 2
        laplacian = i_n + i_s + i_e + i_w
        q = np.sqrt(np.maximum(
            (0.5 * gradient2 - 0.0625 * laplacian ** 2) /
            (img * img + 1e-12), 0.0))
        c = 1.0 / (1.0 + ((q * q - q0 * q0) / (q0 * q0 * (1.0 + q0 * q0) + 1e-12)))
        c = np.clip(c, 0.0, 1.0)
        cp = np.pad(c, 1, mode="edge")
        c_s, c_e = cp[2:, 1:-1], cp[1:-1, 2:]
        img = img + lam * (c * i_n + c_s * i_s + c * i_w + c_e * i_e)
        img = np.clip(img, 1e-6, None)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    return img.astype(np.float32)


def lee_filter(img: np.ndarray, win: int = 7, noise_var: float | None = None) -> np.ndarray:
    """Lee filter for speckle reduction.

    out = mean + k * (img - mean),  k = var_local / (var_local + var_noise)
    In uniform areas k -> 0 (strong smoothing); near edges k -> 1 (preserved).
    """
    img = np.asarray(img, dtype=np.float64)
    mean = cv2.boxFilter(img, -1, (win, win), normalize=True)
    var = np.maximum(cv2.boxFilter(img * img, -1, (win, win), normalize=True) - mean ** 2, 0.0)
    if noise_var is None:
        noise_var = float(np.mean(var))
    k = var / (var + noise_var + 1e-12)
    return (mean + k * (img - mean)).astype(np.float32)


def preprocess_for_segmentation(gray: np.ndarray, size: int = 256) -> np.ndarray:
    """Denoise + resize + normalize to model input (1, size, size)."""
    denoised = lee_filter(np.clip(gray, 0, 1))
    resized = cv2.resize(denoised, (size, size), interpolation=cv2.INTER_AREA)
    return resized[None, :, :].astype(np.float32)
