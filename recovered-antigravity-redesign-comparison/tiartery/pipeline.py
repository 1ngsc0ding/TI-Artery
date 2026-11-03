"""End-to-end TI-Artery pipeline: steps 1-9 + risk integration.

run_pipeline() takes one B-mode image (or the best frame of a video),
runs QC -> denoise -> plaque segmentation -> quantification ->
calibration -> IMT -> risk, and returns a result dict plus an annotated
image for clinician review (human-in-the-loop).
"""
import os

import numpy as np

from . import preprocessing as pp
from . import segmentation as seg
from . import imt as imt_mod
from . import plaque as plq
from . import risk as risk_mod

# Default assumed pixel spacing used when scanner spacing is unknown.
# Typical clinical CCA images: ~0.10–0.15 mm/px at standard zoom.
_ASSUMED_SPACING_MM = 0.10


def _risk_unavailable_reasons(usable: bool, quality: dict, plaque_present,
                              imt_res: dict, clinical: dict) -> list[str]:
    """Return reasons why a risk tier cannot be shown.

    We only block on quality failure or missing plaque model.  IMT being
    a demo estimate is allowed — the tier will display with a warning.
    Clinical inputs are optional (omitted inputs simply don't add to the score).
    """
    reasons = []
    if not usable:
        reasons.extend(pp.quality_issues(quality) or [
            "This frame did not pass image-quality checks."])
    if plaque_present is None:
        reasons.append("Plaque assessment is missing because the plaque model is not loaded.")
    return reasons


def run_pipeline(image_path: str,
                 model: "seg.UNet | None" = None,
                 model_path: str | None = None,
                 pixel_spacing_mm: float | None = None,
                 clinical: dict | None = None,
                 device: str = "cpu",
                 wall_model_path: str | None = None,
                 wall_model=None,
                 plaque_threshold: float = 0.5,
                 plaque_min_area: int = 0,
                 denoise: str = "lee") -> dict:
    clinical = clinical or {}
    if pixel_spacing_mm is not None and (not np.isfinite(pixel_spacing_mm) or pixel_spacing_mm <= 0):
        raise ValueError("pixel_spacing_mm must be finite and positive")

    # Step 1: ultrasound input (still or video)
    if os.path.splitext(image_path)[1].lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        gray, source_meta = pp.load_ultrasound(image_path)
    else:
        gray, source_meta = pp.load_grayscale(image_path), {"source": "image", "n_frames": 1}

    # Step 2: frame selection / QC
    quality = pp.frame_quality(gray)
    usable = pp.is_usable(quality)

    # Step 3: speckle noise reduction (Lee and/or SRAD). Segmentation stays
    # on raw gray so inference matches training.
    if denoise == "srad":
        denoised = pp.srad(gray)
    elif denoise == "both":
        denoised = pp.srad(pp.lee_filter(gray))
    else:
        denoised = pp.lee_filter(gray)

    # Steps 6-7: plaque detection + segmentation
    mask = None
    if model is None and model_path is not None:
        model = seg.load_model(model_path, device)
    if model is not None:
        psize = getattr(model, "input_size", 320)
        mask = seg.predict_mask(model, gray, size=psize, thr=plaque_threshold,
                                min_area=plaque_min_area, device=device)
    plaque_present = bool(plq.detect_plaque(mask)) if mask is not None else None

    # Step 8-9: quantification + calibration
    quantification = plq.quantify(mask, pixel_spacing_mm) if mask is not None else None
    tpa = quantification.get("area_mm2") if quantification else None

    # When no pixel_spacing, estimate plaque dimensions with assumed spacing so
    # the report can show an interpreted plaque size (flagged as estimated).
    if quantification and not quantification.get("calibrated"):
        s = _ASSUMED_SPACING_MM
        qp = quantification
        if qp.get("length_px") is not None:
            quantification["length_mm_est"] = round(float(qp["length_px"]) * s, 2)
            quantification["thickness_mm_est"] = (round(float(qp["thickness_px"]) * s, 2)
                                                   if qp.get("thickness_px") else None)
            quantification["area_mm2_est"] = round(float(qp["area_px"]) * s * s, 2)
            quantification["assumed_spacing_mm"] = s

    # Steps 4-5: vessel-wall U-Net -> LI/MA geometry, falling back to
    # classical-CV demo estimate when no trained wall model is available.
    imt_res = None
    wall = None
    if wall_model is None and wall_model_path is not None:
        wall_model = seg.load_model(wall_model_path, device)

    if wall_model is not None:
        wsize = getattr(wall_model, "input_size", 320)
        wall, wall_prob = seg.predict_mask(
            wall_model, gray, size=wsize, device=device, return_prob=True)
        imt_res = imt_mod.imt_from_wall_mask(wall, pixel_spacing_mm, prob=wall_prob)
        imt_res.pop("li_pts", None)
        imt_res.pop("ma_pts", None)
        # Wall model succeeded but no pixel spacing → estimate mm with assumed spacing
        if not imt_res.get("measurement_failed") and imt_res.get("mean_imt_mm") is None:
            mean_px = imt_res.get("mean_imt_px")
            max_px = imt_res.get("max_imt_px")
            if mean_px is not None:
                imt_res["mean_imt_mm"] = round(float(mean_px) * _ASSUMED_SPACING_MM, 4)
                imt_res["max_imt_mm"] = (round(float(max_px) * _ASSUMED_SPACING_MM, 4)
                                         if max_px is not None else None)
                imt_res["assumed_spacing_mm"] = _ASSUMED_SPACING_MM
                imt_res["demo"] = True
                imt_res["note"] = (
                    f"Wall boundaries from trained model (sub-pixel LI/MA, Mannheim mean/max). "
                    f"Pixel spacing not provided — mm estimated at assumed {_ASSUMED_SPACING_MM} mm/px. "
                    f"Enter scanner pixel spacing for accurate millimetre values."
                )

    # Fallback: classical edge-detection demo estimate (always flagged as DEMO)
    if imt_res is None or imt_res.get("measurement_failed"):
        spacing_used = pixel_spacing_mm if pixel_spacing_mm else _ASSUMED_SPACING_MM
        demo_res = imt_mod.estimate_imt_demo(denoised, spacing_used)
        if imt_res is None:
            demo_res["note"] = (
                "DEMO estimate (no wall model loaded) — classical edge detection, "
                "NOT validated. Provide pixel spacing from your scanner for accuracy."
            )
        else:
            demo_res["note"] = (
                "DEMO estimate (wall model did not produce a measurable segment) — "
                "classical edge detection, NOT validated."
            )
        if pixel_spacing_mm is None:
            demo_res["assumed_spacing_mm"] = _ASSUMED_SPACING_MM
        imt_res = demo_res

    imt_mean = imt_res.get("mean_imt_mm")

    # Risk integration — only block when image quality fails or plaque model absent.
    risk_blockers = _risk_unavailable_reasons(
        usable, quality, plaque_present, imt_res, clinical)
    if risk_blockers:
        risk_res = {
            "tier": "unavailable",
            "score": None,
            "reasons": list(risk_blockers),
            "disclaimer": risk_mod.DISCLAIMER,
        }
    else:
        risk_res = risk_mod.assess_risk(
            imt_mean_mm=imt_mean, tpa_mm2=tpa,
            plaque_present=bool(plaque_present),
            age=clinical.get("age"), sbp=clinical.get("sbp"), ldl=clinical.get("ldl"),
            smoking=clinical.get("smoking", False),
            diabetes=clinical.get("diabetes", False))

    annotated = plq.overlay_mask(gray, mask, color=(0, 255, 0), alpha=0.45)
    if wall is not None:
        annotated = plq.overlay_mask(gray, wall, color=(0, 165, 255),
                                     base=annotated, alpha=0.25)
    annotated = plq.draw_overlay_legend(annotated)

    return {
        "image_path": image_path,
        "source": source_meta,
        "frame_usable": usable,
        "frame_quality": quality,
        "denoised_preview": denoised,
        "plaque_present": plaque_present,
        "quantification": quantification,
        "imt": imt_res,
        "imt_class": imt_mod.classify_imt(imt_mean),
        "risk": risk_res,
        "risk_unavailable_reasons": risk_blockers,
        "annotated_image": annotated,
        "human_in_the_loop": True,
        "disclaimer": risk_mod.DISCLAIMER,
    }
