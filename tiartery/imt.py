"""Step 5 of the TI-Artery pipeline: IMT calculation per Mannheim Consensus.

Design rule (per proposal): the AI only locates the two boundaries
(lumen-intima and media-adventitia interfaces). The IMT value itself is
pure geometry: the distance between the two polylines, reported as
mean and max IMT over the measured segment.

A trained vessel-wall segmentation model slots in via `WallModel`;
until annotated wall data is available, `estimate_imt_demo` provides a
classical-CV fallback that is ALWAYS flagged as unvalidated demo output.
"""
import cv2
import numpy as np


def boundary_distances_px(li_pts: np.ndarray, ma_pts: np.ndarray) -> np.ndarray:
    """Perpendicular-ish distance between two polylines with matched points.

    li_pts / ma_pts: (N, 2) arrays of (x, y) in pixels, resampled to the
    same N points along the vessel axis.
    """
    li = np.asarray(li_pts, dtype=np.float64)
    ma = np.asarray(ma_pts, dtype=np.float64)
    if li.shape != ma.shape:
        raise ValueError("Boundary point arrays must have the same shape")
    return np.linalg.norm(li - ma, axis=1)


def compute_imt(li_pts: np.ndarray, ma_pts: np.ndarray,
                pixel_spacing_mm: float) -> dict:
    """Mean / max IMT in mm from the two wall boundaries."""
    d_mm = boundary_distances_px(li_pts, ma_pts) * pixel_spacing_mm
    return {
        "mean_imt_mm": float(np.mean(d_mm)),
        "max_imt_mm": float(np.max(d_mm)),
        "std_imt_mm": float(np.std(d_mm)),
        "n_points": int(len(d_mm)),
        "pixel_spacing_mm": float(pixel_spacing_mm),
    }


def _far_wall_mask(binary: np.ndarray) -> np.ndarray:
    """Keep the deepest (largest-y) wall component — Mannheim far-wall CCA."""
    num, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if num <= 2:
        return binary
    best, best_y = 1, -1.0
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            continue
        ys = np.flatnonzero((lab == i).any(axis=1))
        if len(ys) == 0:
            continue
        ymean = float(ys.mean())
        if ymean > best_y:
            best, best_y = i, ymean
    return lab == best


def _subpixel_interface(col: np.ndarray, y_int: int, end: str) -> float:
    """Linear 0.5-crossing around a binary edge (sub-pixel Mannheim leading edge)."""
    y_int = int(np.clip(y_int, 0, len(col) - 1))
    if end == "li":
        y0 = max(y_int - 1, 0)
        a, b = float(col[y0]), float(col[y_int])
        if a == b:
            return float(y_int)
        t = (0.5 - a) / (b - a)
        return float(y0 + np.clip(t, 0.0, 1.0))
    y1 = min(y_int + 1, len(col) - 1)
    a, b = float(col[y_int]), float(col[y1])
    if a == b:
        return float(y_int)
    t = (0.5 - a) / (b - a)
    return float(y_int + np.clip(t, 0.0, 1.0))


def interfaces_from_wall_mask(mask: np.ndarray, prob: np.ndarray | None = None,
                              subpixel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Lumen-intima (top) and media-adventitia (bottom) polylines from an IMC mask.

    Matches the CUBS rasterization convention: for each annotated column the
    first wall pixel is LI and the last is MA. Distances are then geometric
    (Mannheim-style mean/max along the far-wall segment), not a network
    regression of IMT.
    """
    binary = _far_wall_mask(mask > 0)
    xs, li_y, ma_y = [], [], []
    for x in range(binary.shape[1]):
        ys = np.flatnonzero(binary[:, x])
        if len(ys) >= 2:
            xs.append(x)
            li, ma = float(ys[0]), float(ys[-1])
            if subpixel:
                col = (prob[:, x] if prob is not None else binary[:, x].astype(np.float64))
                li = _subpixel_interface(col, int(ys[0]), "li")
                ma = _subpixel_interface(col, int(ys[-1]), "ma")
            li_y.append(li)
            ma_y.append(ma)
    if len(xs) < 5:
        return np.zeros((0, 2)), np.zeros((0, 2))
    li = np.stack([np.asarray(xs, dtype=np.float64), np.asarray(li_y)], axis=1)
    ma = np.stack([np.asarray(xs, dtype=np.float64), np.asarray(ma_y)], axis=1)
    return li, ma


def perpendicular_distances_px(li_pts: np.ndarray, ma_pts: np.ndarray) -> np.ndarray:
    """Thickness along the local wall normal (closer to Mannheim than raw |dy|)."""
    li = np.asarray(li_pts, dtype=np.float64)
    ma = np.asarray(ma_pts, dtype=np.float64)
    if len(li) < 3:
        return np.linalg.norm(li - ma, axis=1)
    t = np.stack([np.gradient(li[:, 0]), np.gradient(li[:, 1])], axis=1)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(n_norm, 1e-8)
    return np.abs(((ma - li) * n).sum(axis=1))


def imt_from_wall_mask(mask: np.ndarray, pixel_spacing_mm: float | None = None,
                       subpixel: bool | None = None, prob: np.ndarray | None = None) -> dict:
    """Mean/max IMT from a wall mask. Sub-pixel: linear interpolation of edges."""
    if subpixel is None:
        subpixel = prob is not None
    li, ma = interfaces_from_wall_mask(mask, prob=prob, subpixel=subpixel)
    if len(li) == 0:
        return {"mean_imt_mm": None, "max_imt_mm": None, "demo": False,
                "measurement_failed": True,
                "note": "Wall mask has no measurable LI/MA column pair"}
    d_px = perpendicular_distances_px(li, ma)
    out = {
        "mean_imt_px": float(np.mean(d_px)),
        "max_imt_px": float(np.max(d_px)),
        "std_imt_px": float(np.std(d_px)),
        "n_points": int(len(d_px)),
        "demo": False,
        "measurement_failed": False,
        "note": "Geometry from far-wall LI/MA (sub-pixel, wall-normal; Mannheim mean/max)",
        "li_pts": li,
        "ma_pts": ma,
    }
    if pixel_spacing_mm is not None:
        d_mm = d_px * float(pixel_spacing_mm)
        out.update({
            "mean_imt_mm": float(np.mean(d_mm)),
            "max_imt_mm": float(np.max(d_mm)),
            "std_imt_mm": float(np.std(d_mm)),
            "n_points": int(len(d_mm)),
            "pixel_spacing_mm": float(pixel_spacing_mm),
        })
    return out


def classify_imt(mean_imt_mm: float | None) -> str:
    """Mannheim-flavoured screening bands (not a diagnosis)."""
    if mean_imt_mm is None:
        return "unknown"
    if mean_imt_mm < 0.9:
        return "normal range"
    if mean_imt_mm < 1.5:
        return "thickened"
    return "plaque-range thickening"


def estimate_imt_demo(gray: np.ndarray, pixel_spacing_mm: float | None = None) -> dict:
    """Classical-CV fallback: find the far-wall double line, measure its gap.

    DEMO ONLY - not validated. A trained wall-segmentation model should
    replace this. Returns the measurement plus an explicit demo flag.

    Strategy: tries multiple horizontal ROI bands, uses strong blur to
    suppress speckle, filters gap candidates to the realistic IMT range
    (0.4-4 mm at the assumed pixel spacing), and reports the median.
    """
    if gray.ndim != 2:
        gray = gray[:, :, 0] if gray.ndim == 3 else gray.squeeze()
    h, w = gray.shape
    spacing = float(pixel_spacing_mm) if pixel_spacing_mm else 0.10

    # Typical IMT: 0.4–4.0 mm  →  pixel gap range
    gap_min = max(3, int(0.4 / spacing))
    gap_max = min(h // 4, int(4.0 / spacing) + 10)

    # Try several ROI bands: lower 60%, lower 40%, middle 50%
    roi_slices = [
        (int(h * 0.35), h),        # lower 65%
        (int(h * 0.45), h),        # lower 55%
        (int(h * 0.20), int(h * 0.70)),  # middle band
        (0, h),                    # full image fallback
    ]

    best_gaps = []
    for r_start, r_end in roi_slices:
        if r_end - r_start < 20:
            continue
        roi = gray[r_start:r_end, :]
        u8 = (np.clip(roi, 0, 1) * 255).astype(np.uint8)
        # Stronger blur to suppress speckle noise
        blurred = cv2.GaussianBlur(u8, (0, 0), 3.0)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.abs(gy)

        gaps = []
        # Sample every 2nd column for better coverage
        for x in range(0, w, 2):
            col = edges[:, x]
            peaks, _ = _find_peaks(col, min_dist=4)
            if len(peaks) < 2:
                continue
            # two strongest peaks = LI and MA interfaces
            order = np.argsort(col[peaks])[::-1][:2]
            y = sorted(peaks[order])
            gap = abs(y[1] - y[0])
            if gap_min <= gap <= gap_max:
                gaps.append(gap)

        if len(gaps) >= 10:          # enough votes → use this ROI
            best_gaps = gaps
            break
        if len(gaps) > len(best_gaps):
            best_gaps = gaps         # keep best so far

    # Second-pass fallback: use first-vs-last significant edge per column
    if len(best_gaps) < 5:
        roi = gray[int(h * 0.35):, :]
        u8 = (np.clip(roi, 0, 1) * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(u8, (0, 0), 4.0)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.abs(gy)
        col_max = float(edges.max()) if edges.max() > 0 else 1.0
        thresh = col_max * 0.25
        for x in range(0, w, 2):
            col = edges[:, x]
            sig = np.flatnonzero(col > thresh)
            if len(sig) < 2:
                continue
            gap = int(sig[-1]) - int(sig[0])
            if gap_min <= gap <= gap_max:
                best_gaps.append(gap)

    if not best_gaps:
        return {"mean_imt_mm": None, "max_imt_mm": None, "demo": True,
                "measurement_failed": False,
                "note": "No wall double-line found by edge detection."}

    gap_px = float(np.median(best_gaps))
    gap_px_p90 = float(np.percentile(best_gaps, 90))
    return {
        "mean_imt_mm": round(gap_px * spacing, 4),
        "max_imt_mm": round(gap_px_p90 * spacing, 4),
        "gap_px": gap_px,
        "demo": True,
        "assumed_spacing_mm": None if pixel_spacing_mm else spacing,
        "note": "DEMO estimate from classical edge detection — NOT validated. "
                "Replace with a trained wall-segmentation model for clinical use.",
    }


def _find_peaks(col: np.ndarray, min_dist: int = 3):
    """Tiny local-maximum finder to avoid a scipy dependency."""
    peaks = []
    last = -min_dist - 1
    for i in range(1, len(col) - 1):
        if col[i] > col[i - 1] and col[i] >= col[i + 1] and i - last > min_dist:
            peaks.append(i)
            last = i
    return np.array(peaks), None
