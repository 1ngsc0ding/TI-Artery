"""Risk integration: combine imaging findings with clinical risk factors.

Transparent, rule-based screening support - NOT a diagnosis. The score
simply counts elevated findings so a clinician can see exactly why a
case was flagged. Final decision always stays with the physician
(human-in-the-loop, per proposal).
"""
DISCLAIMER = ("Unvalidated rule-based screening score, not a stroke probability or diagnosis. "
              "A qualified physician must review every result.")


def assess_risk(imt_mean_mm: float | None,
                tpa_mm2: float | None,
                plaque_present: bool | None,
                age: int | None = None,
                sbp: float | None = None,
                ldl: float | None = None,
                smoking: bool | None = None,
                diabetes: bool | None = None) -> dict:
    missing = []
    if imt_mean_mm is None:
        missing.append("A calibrated IMT measurement is unavailable.")
    if tpa_mm2 is None:
        missing.append("Calibrated plaque area is unavailable.")
    if plaque_present is None:
        missing.append("Plaque assessment is unavailable.")
    for value, label in ((age, "Age"), (sbp, "Systolic blood pressure"),
                         (ldl, "LDL"), (smoking, "Smoking status"),
                         (diabetes, "Diabetes status")):
        if value is None:
            missing.append(f"{label} was not provided.")
    if missing:
        return {"tier": "unavailable", "score": None, "reasons": missing,
                "disclaimer": DISCLAIMER}

    score, reasons = 0, []

    # --- imaging findings ---
    if plaque_present:
        score += 2
        reasons.append("plaque detected on B-mode")
    if imt_mean_mm is not None and imt_mean_mm >= 0.9:
        score += 1
        reasons.append(f"IMT {imt_mean_mm:.2f} mm >= 0.9 mm (thickened)")
    if tpa_mm2 is not None and tpa_mm2 >= 60:
        score += 1
        reasons.append(f"total plaque area {tpa_mm2:.1f} mm^2 elevated")

    # --- clinical factors ---
    if age is not None and age >= 60:
        score += 1
        reasons.append(f"age {age}")
    if sbp is not None and sbp >= 140:
        score += 1
        reasons.append(f"systolic BP {sbp:.0f} mmHg (hypertension)")
    if ldl is not None and ldl >= 130:
        score += 1
        reasons.append(f"LDL {ldl:.0f} mg/dL elevated")
    if smoking:
        score += 1
        reasons.append("smoking")
    if diabetes:
        score += 1
        reasons.append("diabetes")

    tier = "low" if score <= 1 else ("elevated" if score <= 3 else "high")
    return {
        "tier": tier,
        "score": score,
        "reasons": reasons,
        "disclaimer": DISCLAIMER,
    }
