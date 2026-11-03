# TI-Artery

TI-Artery is a research prototype for viewing carotid B-mode ultrasound images, producing plaque and vessel-wall segmentation overlays, measuring plaque geometry and intima-media thickness (IMT), and displaying an unvalidated rule-based screening score. It is not a medical device, diagnosis, or stroke-probability model. Outputs require qualified clinical review.

## Current scope

- The Gradio interface accepts a still B-mode image and optional patient values and scanner pixel spacing.
- Separate U-Net checkpoints produce plaque and wall masks. Geometry code derives plaque dimensions and IMT from those masks.
- Millimetres and mm² require valid scanner calibration. Without it, the main report shows only available pixel measurements, suppresses assumed-scale/demo values and size-based recommendations, and marks risk assessment unavailable with a calibration reason.
- Video helpers exist in the pipeline, but the current UI accepts still images only.
- The rule-based score has no clinical outcome validation; it must not be interpreted as stroke probability.

## Setup and run

Python 3.10+ is recommended. Install a PyTorch build matching the machine (CPU or the installed CUDA version) using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The model weights are deliberately not stored in this Git repository. The app looks for `models/plaque_finetune.pt` and `models/wall_unet.pt` (see `app.py` for fallback names). Obtain the matching, authorized checkpoint files from the project maintainer and place them at those paths before expecting the trained segmentation results. Without them, plaque output may be unavailable and the wall measurement may fall back to an explicitly marked demo estimator.

Run the local prototype:

```bash
python app.py
```

Then open `http://localhost:7860`. The app automatically selects CUDA when PyTorch reports an available GPU, otherwise CPU; `TIARTERY_DEVICE` can explicitly select a device. `TIARTERY_THRESHOLD` defaults to `0.5`; `TIARTERY_MIN_AREA` defaults to `0`. Do not change these values to imply clinical performance.

## Tests

Run the repository test suite with:

```bash
python -m pytest -q
```

Tests exercise workflow and measurement/risk behavior. Passing tests establish code-path properties only; they do not establish clinical accuracy.

## Data and model artifacts

Training/evaluation source is included, but datasets, images, checkpoints, training runs, logs, and generated evaluation overlays are excluded by `.gitignore` to avoid publishing patient data and large artifacts. The existing main checkpoints are approximately 31 MB each; the retained last-training checkpoint is approximately 93 MB. For a later Hugging Face deployment, store versioned weights in a Hugging Face Hub model repository (private or access-controlled if required), or use Git LFS for model files if GitHub storage is preferred. Do not put datasets or patient images in a public repository. Model downloading is not automated by this code snapshot.

The last stored evaluation is in `evaluation/accuracy_20261030/REPORT.md` and `metrics.json`; generated images from that run are ignored. It reports plaque test global Dice 0.6682 / IoU 0.5017 on an exploratory split, with image-level false positives on 11/24 plaque-negative images. Wall test after removing two exact train/test image duplicates reports global Dice 0.7446 / IoU 0.5932 on 100 images. IMT MAE was 0.2740 mm against CUBS profile-derived references with per-image calibration. Patient-level independence is not established for either dataset; the plaque test was used for model selection. These segmentation/measurement comparisons do not validate risk scoring or stroke predictions. See `PROJECT_SUMMARY_TH.md` for the evidence-based project summary.

## Limitations

- The plaque test split is exploratory, and no complete patient-level split provenance is available. Two exact duplicate CUBS images were found across wall train/test; they are excluded from the reported clean test result.
- The wall checkpoint omits `resize_mode`. The checked-in trainer defaults to letterbox, but the checkpoint validation Dice is reproduced by stretch (0.762632 vs 0.725780 for letterbox); the saved training invocation is missing. Current inference uses a consistent stretch transform for both input and output restoration. Held-out comparison, boundary overlays, and the severe sample 062 IMT error are in `evaluation/wall_geometry_trace_20261030/REPORT.md`.
- No labelled stroke outcomes or labelled clinical risk decisions are present. Dice, IoU, false-positive counts, and IMT MAE cannot establish clinical risk accuracy.
- The score is an unvalidated rule-based screening score, not a stroke probability or diagnosis.

## Key files

- `app.py`: Gradio interface and checkpoint selection.
- `tiartery/pipeline.py`: image processing, segmentation, measurement, calibration, and risk integration.
- `tiartery/risk.py`: transparent screening-score rules.
- `train_plaque_unet.py`, `train_wall_unet.py`: training code (not run by the app).
- `tests/`: regression and workflow tests.
- `PROJECT_SUMMARY_TH.md`: detailed Thai summary of current code, datasets, checkpoints, and existing evaluation evidence.
# TI_Artery
