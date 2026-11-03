# Recovered Antigravity redesign candidate — 2026-10-30

This is an isolated source snapshot. The active projects were not changed.

## Provenance

- `app.py`, `tiartery/pipeline.py`, and `tiartery/report.py` were extracted from complete Antigravity conversation-database file-write payloads (steps 24, 20, and 22). Recorded subsequent edits were replayed verbatim: pipeline steps 118, 148, 170; report step 172.
- `tiartery/imt.py`, `risk.py`, `plaque.py`, and `preprocessing.py` were extracted from complete Antigravity file-view payloads; IMT step 116 was replayed verbatim.
- The session contains no edits to `tiartery/segmentation.py`, `geometry.py`, `__init__.py`, or `requirements.txt`; those supporting files were copied byte-for-byte from local baseline commit `cdaefd6a5720329c380bacd2c5f7acc54d678878`. Their exact initial session snapshots were not logged as file contents.
- Checkpoints and `samples/demo.png` were copied byte-for-byte from the Antigravity project path `/home/numing/carotid_artery_project/ti-artery-gpu`.
- The matching recorded browser screenshots and notes are in the sibling `evidence-antigravity-session-20261030` folder.

## Original assumed-spacing behavior

The recovered code defines `_ASSUMED_SPACING_MM = 0.10`. When spacing is blank/zero, the demo path derives IMT and plaque millimetres from pixels using that assumed scale. The estimated IMT value is passed to the rule scorer and can add its IMT point; estimated plaque area can also add its point if it crosses the area threshold. These are assumed-scale demo estimates, not scanner-calibrated values.

The matching recorded browser result was mean/max IMT `2.630/3.289 mm`, estimated plaque area `53.2 mm²`, and `4/8 — High Risk`. With the shown values age 65, SBP 130, LDL 110, smoking off and diabetes off, the points are plaque +2, assumed-scale IMT +1, age +1; area is below 60 mm² and the remaining factors add 0. The result is a rule-based demonstration output, not validated clinical accuracy or a patient-specific stroke probability.
