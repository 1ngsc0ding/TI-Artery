# Wall inference and IMT trace (2026-10-30)

This is a focused inference comparison on CUBS held-out images, not an evaluation of the user's Space upload and not a clinical validation. Images 083, 086, and 062 have corresponding wall masks, manual LI/MA annotations, per-image scanner calibration, and reference IMT in ../data/cubs/manifest.json.

## Execution path traced

The app accepts a Gradio NumPy image, converts it to grayscale, and writes a temporary PNG (app.py:88-105). That conversion does not read scanner metadata. The pipeline loads the grayscale PNG, keeps the raw grayscale image for model inference, and uses the denoised copy only for preview/demo fallback (tiartery/pipeline.py:73-95). The wall model loads models/wall_unet.pt; the checkpoint is a 320-pixel U-Net, base 32, threshold 0.5, validation Dice 0.7626315, and has no resize_mode field.

For each inference, predict_prob resizes the raw image, predicts a probability map, restores it to original image dimensions, and predict_mask thresholds at 0.5 (tiartery/segmentation.py:226-267). With stretch, the output restoration resizes the complete map to the original shape; with letterbox, it removes the exact padding computed by the matching resize routine before resizing. The current loader now resolves absent mode metadata to stretch, and preprocessing and restoration both normalize the same mode (tiartery/geometry.py, tiartery/segmentation.py:290-300).

IMT extraction then retains the selected far-wall component, takes the first and last positive mask pixels in each usable column as LI and MA, adjusts the boundaries to the 0.5 probability crossing, and computes wall-normal distance in pixels (tiartery/imt.py:20-102). The scanner calibration is multiplied in only after pixel distances have been obtained. Thus a lack of calibration prevents conversion to mm; it does not by itself prevent a pixel measurement.

## Resize-mode evidence

The checkpoint does not record training resize mode. Although the checked-in trainer defaults to letterbox, the old accuracy script read the checkpoint's absent mode as stretch while its text described the calculation as letterbox. Repeating the 103-image validation calculation explicitly gives:

| Mode | Global Dice | Global IoU | Mean image Dice | Mean image IoU |
|---|---:|---:|---:|---:|
| Stretch | 0.762632 | 0.616333 | 0.715896 | 0.570884 |
| Letterbox | 0.725780 | 0.569588 | 0.681620 | 0.530429 |

Stretch reproduces the checkpoint's stored validation Dice to the reported precision. This is evidence that the checkpoint's validation/inference path used stretch, but because the training invocation/config was not saved, it does not prove what preprocessing was used during training. The actual old loader already supplied "stretch" for this checkpoint; it did not return None. Therefore the previously reported current-app letterbox restoration mismatch was a mistaken diagnosis. The code change here makes mode normalization explicit and keeps input resize and output restoration paired; it does not claim an accuracy gain or change the current checkpoint's inference geometry.

## Held-out comparison

Boundary error is the mean absolute vertical difference of predicted and mask-derived LI and MA points over common columns. IMT error compares predicted-mask geometry with the manifest's profile-derived gt_imt_mm, using each image's manifest cf_mm_px. The profile reference is related to, but not independent from, the wall labels.

| CUBS test image | Mode | Wall Dice | Wall IoU | LI/MA boundary MAE (px) | Predicted IMT (px) | Predicted IMT (mm) | Reference IMT (mm) | Absolute IMT error (mm) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 083 | Stretch | 0.9087 | 0.8327 | 1.579 | 16.508 | 1.0318 | 0.9631 | 0.0687 |
| 083 | Letterbox | 0.7557 | 0.6073 | 2.829 | 21.190 | 1.3243 | 0.9631 | 0.3613 |
| 086 | Stretch | 0.8875 | 0.7978 | 2.372 | 26.721 | 1.2292 | 1.1376 | 0.0916 |
| 086 | Letterbox | 0.8680 | 0.7667 | 3.299 | 28.249 | 1.2995 | 1.1376 | 0.1619 |
| 062 | Stretch | 0.3392 | 0.2042 | 7.711 | 18.762 | 1.1727 | 0.3041 | 0.8686 |
| 062 | Letterbox | 0.2935 | 0.1720 | 8.571 | 20.118 | 1.2574 | 0.3041 | 0.9533 |

The matching mask-derived reference measurements are 15.215 px / 0.9510 mm for 083, 23.913 px / 1.1000 mm for 086, and 4.864 px / 0.3040 mm for 062. The extraction and calibration reproduce the reference closely on the reference masks; on 062, the predicted segmentation is much too large (9,519 predicted wall pixels versus 1,944 reference pixels), and the resulting pixel IMT is 18.762 px. This is a segmentation failure, not a missing-calibration failure or an LI/MA conversion error. The pixel measurement exists but is wrong because the predicted wall mask is wrong.

## Visual evidence

Overlays show reference LI/MA boundaries in green/yellow and predicted LI/MA in red/magenta. These are saved at the original image dimensions.

- Good segmentation example: [083, checkpoint-compatible stretch](083_checkpoint_compatible_stretch_boundaries.png) and [083, letterbox alternative](083_letterbox_alternative_boundaries.png).
- Failure example: [062, checkpoint-compatible stretch](062_checkpoint_compatible_stretch_boundaries.png) and [062, letterbox alternative](062_letterbox_alternative_boundaries.png).
- Full per-image values and boundary coordinate diagnostics: [metrics.json](metrics.json).

The overlay generator originally reversed (x,y) coordinates when drawing and thereby made boundaries appear transposed. It now draws OpenCV coordinates in their correct order. The figures here are regenerated with the corrected plotting code.

## Calibration and limits

Running the production run_pipeline() path on CUBS 083 with its manifest calibration returns mean IMT 16.508 px / 1.0318 mm. Running it with pixel_spacing_mm=None still returns 16.508 px, sets calibrated mm to None, stores a separate demo estimate, and leaves risk unavailable. The app does not extract DICOM pixel spacing or read a visible ruler; it converts the uploaded array into a metadata-free temporary PNG. No actual user-uploaded image was available locally, so neither its boundaries nor any visible calibration mark could be inspected. Do not use the CUBS calibration for that upload.

This comparison establishes behavior on three labelled CUBS images only. The exact Space upload remains unassessed. The model's wall segmentation can fail, as 062 demonstrates; no clinical accuracy or stroke-risk accuracy is established. No model was retrained and no thresholds were changed.
