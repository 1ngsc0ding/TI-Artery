"""TI-Artery: AI for carotid IMT + plaque analysis from B-mode ultrasound.

Pipeline steps (per project proposal):
 1. Ultrasound input          5. IMT calculation (Mannheim)
 2. Frame selection (QC)      6. Plaque detection
 3. Speckle noise reduction   7. Plaque segmentation (U-Net)
 4. Vessel wall segmentation  8. Plaque quantification (planimetry)
                              9. Calibration (pixel^2 -> mm^2)
 + Risk integration (clinical factors) -> screening support output.
"""

__version__ = "0.1.0"
