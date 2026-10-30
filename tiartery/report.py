"""Plain-English result cards for the Gradio review UI.

Formatting only: does not change model predictions.
"""
from __future__ import annotations

import html
import json

from .preprocessing import quality_issues
from .risk import DISCLAIMER

SCOPE = ("CCA B-mode screening only. This tool does not measure blood-flow "
         "stenosis, plaque volume in 3-D, or whether a plaque is likely to rupture.")


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _card(title: str, body: str, accent: str = "#4f46e5") -> str:
    return (
        f'<div style="border-left:4px solid {accent};padding:16px 18px;'
        f'margin:0 0 0 0;background:linear-gradient(135deg,#1e1b4b 0%,#312e81 100%);'
        f'border-radius:8px;">'
        f'<div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;'
        f'color:#a5b4fc;margin-bottom:10px;font-weight:700;">{_esc(title)}</div>'
        f'{body}</div>'
    )


def _p(text: str, *, strong: bool = False, color: str = "#e0e7ff") -> str:
    weight = "font-weight:700;" if strong else "font-weight:400;"
    return f'<p style="margin:0 0 8px 0;line-height:1.5;{weight}color:{color};">{_esc(text)}</p>'


def _badge(text: str, color: str, bg: str) -> str:
    return (f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
            f'background:{bg};color:{color};font-size:13px;font-weight:700;'
            f'letter-spacing:.03em;">{_esc(text)}</span>')


def overlay_legend_html() -> str:
    return (
        '<p style="margin:0 0 10px 0;line-height:1.45;font-size:13px;color:#c7d2fe;">'
        '<strong style="color:#e0e7ff;">Overlay:</strong> '
        '<span style="color:#4ade80;font-weight:700;">■ green = plaque</span>&nbsp; '
        '<span style="color:#fb923c;font-weight:700;">■ orange = intima-media wall</span>.'
        '</p>'
    )


def _tier_badge(tier: str | None) -> str:
    if tier is None or tier == "unavailable":
        return _badge("Risk assessment unavailable", "#fff", "#6b7280")
    if tier == "low":
        return _badge("Low screening tier", "#14532d", "#86efac")
    if tier == "elevated":
        return _badge("Elevated screening tier", "#713f12", "#fde68a")
    return _badge("High screening tier", "#fff", "#ef4444")


def _imt_badge(mean_mm: float | None, imt_class: str | None) -> str:
    if mean_mm is None:
        return _badge("IMT unavailable", "#fff", "#6b7280")
    cls = (imt_class or "").lower()
    if "normal" in cls:
        return _badge(f"IMT {mean_mm:.2f} mm — Normal", "#14532d", "#86efac")
    if "thickened" in cls and "plaque" not in cls:
        return _badge(f"IMT {mean_mm:.2f} mm — Thickened", "#713f12", "#fde68a")
    return _badge(f"IMT {mean_mm:.2f} mm — Plaque-range", "#fff", "#ef4444")


def _classify_plaque_length(length_mm: float) -> tuple[str, str, str]:
    """Return (label, text_color, bg_color) for a plaque length in mm."""
    if length_mm < 10:
        return "Small focal plaque", "#14532d", "#86efac"
    if length_mm < 20:
        return "Moderate plaque", "#713f12", "#fde68a"
    return "Extensive plaque", "#fff", "#ef4444"


def _plaque_length_badge(length_mm: float) -> str:
    label, tc, bg = _classify_plaque_length(length_mm)
    return _badge(f"{length_mm:.1f} mm — {label}", tc, bg)


def summary_card(res: dict, clinician_reviewed: bool = False) -> str:
    """Single combined card: IMT + stroke risk + key caveats."""
    imt = res.get("imt") or {}
    mean_mm = imt.get("mean_imt_mm")
    max_mm = imt.get("max_imt_mm")
    imt_class = res.get("imt_class") or "unknown"
    is_demo = imt.get("demo", False)
    assumed_spacing = imt.get("assumed_spacing_mm")
    mean_mm_est = imt.get("mean_imt_mm_est")
    max_mm_est = imt.get("max_imt_mm_est")
    mean_px = imt.get("mean_imt_px")
    max_px = imt.get("max_imt_px")

    risk = res.get("risk") or {}
    tier = risk.get("tier")
    score = risk.get("score")
    risk_reasons = risk.get("reasons") or []

    plaque_present = res.get("plaque_present")
    quality_ok = res.get("frame_usable", True)

    # ---- IMT section ----
    body = '<div style="margin-bottom:16px;">'
    body += '<div style="font-size:12px;color:#a5b4fc;margin-bottom:6px;font-weight:700;">INTIMA-MEDIA THICKNESS (IMT)</div>'

    if mean_mm is not None:
        body += f'<div style="margin-bottom:8px;">{_imt_badge(mean_mm, imt_class)}</div>'
        detail = f"Mean IMT: {mean_mm:.3f} mm"
        if max_mm is not None:
            detail += f"  |  Max IMT: {max_mm:.3f} mm"
        body += _p(detail, color="#c7d2fe")
        if is_demo:
            warn = "⚠ DEMO estimate (classical edge detection, not a trained wall model)"
            if assumed_spacing:
                warn += f" — assumed pixel spacing {assumed_spacing:.3f} mm/px"
            body += _p(warn, color="#fbbf24")
    else:
        body += _p("Calibrated IMT in millimetres is unavailable.", strong=True, color="#f87171")
        if mean_px is not None:
            px_detail = f"Uncalibrated IMT: mean {mean_px:.2f} px"
            if max_px is not None:
                px_detail += f"  |  max {max_px:.2f} px"
            body += _p(px_detail, color="#c7d2fe")
        if mean_mm_est is not None:
            est_detail = f"Demo estimate: mean {mean_mm_est:.3f} mm (not used for scoring)"
            if max_mm_est is not None:
                est_detail += f"  |  max {max_mm_est:.3f} mm"
            body += _p(est_detail, color="#fbbf24")
        if imt.get("note"):
            body += _p(str(imt["note"]), color="#fbbf24")
    body += '</div>'

    # ---- Plaque Size section ----
    q = res.get("quantification") or {}
    plaque_present = res.get("plaque_present")
    length_mm = q.get("length_mm") or q.get("length_mm_est")
    thick_mm = q.get("thickness_mm") or q.get("thickness_mm_est")
    area_mm2 = q.get("area_mm2") or q.get("area_mm2_est")
    is_est = not q.get("calibrated") and q.get("assumed_spacing_mm") is not None

    body += '<div style="margin-bottom:16px;">'
    body += '<div style="font-size:12px;color:#a5b4fc;margin-bottom:6px;font-weight:700;">PLAQUE SIZE</div>'

    if plaque_present is False:
        body += _p("No plaque segmented on this frame.", color="#86efac")
    elif plaque_present is None:
        body += _p("Plaque model not loaded — size unavailable.", color="#6b7280")
    elif length_mm is not None:
        body += f'<div style="margin-bottom:8px;">{_plaque_length_badge(length_mm)}</div>'
        details = f"Length: {length_mm:.1f} mm"
        if thick_mm is not None:
            details += f"  |  Thickness: {thick_mm:.1f} mm"
        if area_mm2 is not None:
            details += f"  |  Area: {area_mm2:.1f} mm\u00b2"
        body += _p(details, color="#c7d2fe")
        # Clinical interpretation of length
        if length_mm < 10:
            interp = "Small focal lesion — typically lower total atherosclerotic burden."
        elif length_mm < 20:
            interp = "Moderate plaque length — warrants clinical follow-up and risk management."
        else:
            interp = "Extensive plaque — associated with elevated cardiovascular risk; refer for specialist evaluation."
        body += _p(f"\u00bb {interp}", color="#c7d2fe")
        if is_est:
            body += _p(f"\u26a0 Length estimated at assumed pixel spacing {q['assumed_spacing_mm']:.3f} mm/px — enter scanner pixel spacing for accuracy.",
                       color="#fbbf24")
    else:
        body += _p("Plaque detected but size could not be measured.", color="#fbbf24")

    body += '</div>'

    # ---- Risk section ----
    body += '<div style="margin-bottom:14px;">'
    body += '<div style="font-size:12px;color:#a5b4fc;margin-bottom:6px;font-weight:700;">UNVALIDATED RULE-BASED SCREENING SCORE</div>'

    if tier and tier != "unavailable":
        body += f'<div style="margin-bottom:8px;">{_tier_badge(tier)}</div>'
        if score is not None:
            body += _p(f"Score: {score} out of 8 possible points", color="#c7d2fe")
        # plaque finding
        if plaque_present is True:
            body += _p("✓ Plaque detected on B-mode", color="#f87171")
        elif plaque_present is False:
            body += _p("✓ No plaque segmented on this frame", color="#86efac")
        # other reasons
        extras = [r for r in risk_reasons if "plaque" not in r.lower()]
        if extras:
            items = "".join(
                f'<li style="color:#c7d2fe;margin-bottom:2px;">{_esc(r)}</li>'
                for r in extras)
            body += f'<ul style="margin:4px 0 8px 18px;padding:0;line-height:1.5;">{items}</ul>'
    else:
        reasons = res.get("risk_unavailable_reasons") or risk_reasons
        body += f'<div style="margin-bottom:8px;">{_tier_badge(None)}</div>'
        if reasons:
            items = "".join(
                f'<li style="color:#fbbf24;margin-bottom:2px;">{_esc(r)}</li>'
                for r in reasons)
            body += f'<ul style="margin:4px 0 8px 18px;padding:0;line-height:1.5;">{items}</ul>'

    body += '</div>'

    # ---- Quality + disclaimer ----
    if not quality_ok:
        body += _p("⚠ Image quality check failed — measurements may be unreliable.", color="#f87171")

    review_txt = ("✓ Marked as reviewed by operator." if clinician_reviewed
                  else "⚠ Pending physician review — do not act on these results alone.")
    body += _p(review_txt, color="#a5b4fc")
    body += _p(DISCLAIMER, color="#818cf8")

    accent = "#ef4444" if tier == "high" else ("#f59e0b" if tier == "elevated" else "#4ade80" if tier == "low" else "#6b7280")
    return _card("Analysis Summary", body, accent=accent)


def technical_log(res: dict) -> str:
    payload = {
        "source": res.get("source"),
        "frame_usable": res.get("frame_usable"),
        "frame_quality": res.get("frame_quality"),
        "plaque_present": res.get("plaque_present"),
        "quantification": res.get("quantification"),
        "imt": {k: v for k, v in (res.get("imt") or {}).items()
                if k not in {"li_pts", "ma_pts"}},
        "imt_class": res.get("imt_class"),
        "risk": res.get("risk"),
        "risk_unavailable_reasons": res.get("risk_unavailable_reasons"),
        "human_in_the_loop": res.get("human_in_the_loop"),
    }
    return json.dumps(payload, indent=2, default=str)


def empty_view(message: str) -> dict:
    body = _p(message, strong=True, color="#f87171")
    body += _p("Upload a B-mode still image and click Analyze.", color="#a5b4fc")
    return {
        "legend": overlay_legend_html(),
        "summary": _card("Analysis Summary", body, accent="#6b7280"),
        "technical": message,
    }


def format_view(res: dict, clinician_reviewed: bool = False) -> dict:
    return {
        "legend": overlay_legend_html(),
        "summary": summary_card(res, clinician_reviewed),
        "technical": technical_log(res),
    }
