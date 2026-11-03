---
title: TI-Artery
emoji: 🫀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
---

# TI-Artery

Research prototype for carotid B-mode image review. It displays plaque and wall overlays, IMT and plaque measurements, and a heuristic screening tier. It is not a medical device and does not diagnose stroke.

## Demo estimate limitations

When scanner pixel spacing is absent, this recovered demo uses an assumed 0.10 mm/px scale for estimated IMT and plaque dimensions. The displayed rule-based screening tier can include those estimates. These are **demo estimates**, not scanner-calibrated patient measurements. The score is an unvalidated heuristic, not a patient-specific stroke probability. A clinician must review every result.

## Runtime

This Space uses the original recovered Antigravity redesign and its two original checkpoints. On ZeroGPU, the Analyze callback is wrapped with `@spaces.GPU`, and model weights are loaded to CUDA at module startup; no paid hardware setting is changed by this source package. Local runs default to CPU unless `TIARTERY_DEVICE` is set.

Required model files: `models/wall_unet.pt` and `models/plaque_finetune.pt`.

## Local setup

Install PyTorch appropriate to your machine, then install the remaining packages from `requirements.txt`. For a GitHub clone without weights, download the two Space files with: `hf download 1ngsm0del/TI-Artery models/wall_unet.pt models/plaque_finetune.pt --repo-type space --local-dir .`

Start with `python app.py`; the app defaults to port 7860 (or uses the `PORT` environment variable).

## Provenance

`RECOVERY_PROVENANCE.md` records which files were recovered from Antigravity history and which support files came from the recorded baseline commit. The checkpoints are preserved byte-for-byte from the recovered version.
