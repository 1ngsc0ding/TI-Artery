"""Steps 6, 8, 9 of the TI-Artery pipeline: plaque detection,
quantification (planimetry principle) and calibration.

The AI draws the boundary (mask). Everything else is geometry:
area = pixel count inside the boundary, converted to mm^2 with the
image's pixel spacing - the automated equivalent of manual planimetry.
"""
import cv2
import numpy as np


def detect_plaque(mask: np.ndarray, min_area_px: int = 50) -> bool:
    """Plaque present if the segmented area exceeds a noise floor."""
    return int((mask > 0).sum()) >= min_area_px


def quantify(mask: np.ndarray, pixel_spacing_mm: float | None = None) -> dict:
    """Measure plaque area/length from a binary mask (uint8, 0/255)."""
    binary = (mask > 0).astype(np.uint8)
    area_px = int(binary.sum())

    length_px, thickness_px = None, None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        (_, _), (bw, bh), _ = cv2.minAreaRect(cnt)
        length_px, thickness_px = float(max(bw, bh)), float(min(bw, bh))

    out = {
        "area_px": area_px,
        "length_px": length_px,
        "thickness_px": thickness_px,
        "calibrated": pixel_spacing_mm is not None,
    }
    if pixel_spacing_mm is not None:
        s = float(pixel_spacing_mm)
        out.update({
            "area_mm2": area_px * s * s,          # Total Plaque Area (TPA)
            "length_mm": length_px * s if length_px else None,
            "thickness_mm": thickness_px * s if thickness_px else None,
            "pixel_spacing_mm": s,
        })
    else:
        out["note"] = ("uncalibrated: pixel spacing unknown, "
                       "areas are in px^2 not mm^2")
    return out


def overlay_mask(gray: np.ndarray, mask: np.ndarray,
                 color: tuple = (0, 255, 0),
                 base: np.ndarray | None = None,
                 alpha: float = 0.45) -> np.ndarray:
    """Draw a translucent overlay on the B-mode image for clinician review."""
    if base is None:
        rgb = cv2.cvtColor((np.clip(gray, 0, 1) * 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)
    else:
        rgb = base.copy()
    if mask is None:
        return rgb
    colored = np.zeros_like(rgb)
    colored[mask > 0] = color
    return cv2.addWeighted(rgb, 1.0, colored, alpha, 0)


def draw_overlay_legend(bgr: np.ndarray) -> np.ndarray:
    """Add a colour key above the review image without covering mask pixels."""
    h, w = bgr.shape[:2]
    bar_h = 36
    bar = np.full((bar_h, w, 3), 24, dtype=bgr.dtype)
    rows = (((0, 255, 0), "Plaque"), ((0, 165, 255), "Wall (intima-media)"))
    x = 10
    for color, label in rows:
        cv2.rectangle(bar, (x, 10), (x + 16, 26), color, -1)
        cv2.putText(bar, label, (x + 22, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        x += 200 if w >= 360 else 160
    return np.vstack([bar, bgr])
