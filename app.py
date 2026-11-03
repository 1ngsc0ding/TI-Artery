"""TI-Artery Gradio demo: B-mode image -> plaque overlay, IMT, stroke risk.

Research prototype only — not a medical device. A clinician must review every result.
"""
import os
import tempfile

import cv2
import gradio as gr

from tiartery import pipeline as pipe
from tiartery import report
from tiartery import segmentation as seg

ROOT = os.path.dirname(os.path.abspath(__file__))


def _first_existing(*names):
    for name in names:
        path = os.path.join(ROOT, "models", name)
        if os.path.exists(path):
            return path
    return os.path.join(ROOT, "models", names[0])


WALL_PATH = os.environ.get("TIARTERY_WALL_MODEL", _first_existing("wall_unet.pt", "my_wall.pt"))
MODEL_PATH = os.environ.get(
    "TIARTERY_PLAQUE_MODEL",
    _first_existing("plaque_finetune.pt", "plaque_unet.pt", "originals/trial_0015_best.pt"))
THRESHOLD = float(os.environ.get("TIARTERY_THRESHOLD", "0.5"))
MIN_AREA = int(os.environ.get("TIARTERY_MIN_AREA", "0"))
DEVICE = os.environ.get("TIARTERY_DEVICE", "cpu")

_plaque = seg.load_model(MODEL_PATH, DEVICE) if os.path.exists(MODEL_PATH) else None
_wall_path = WALL_PATH if os.path.exists(WALL_PATH) else None
if _plaque is None:
    print(f"[warn] plaque model missing: {MODEL_PATH}")
if _wall_path is None:
    print(f"[warn] wall model missing: {WALL_PATH} — will use demo IMT fallback")


def analyze(image, age, sbp, ldl, smoking, diabetes, pixel_spacing, clinician_reviewed):
    if image is None:
        v = report.empty_view("Please upload a B-mode still image.")
        return None, v["summary"]

    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    path = handle.name
    handle.close()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    cv2.imwrite(path, gray)

    try:
        res = pipe.run_pipeline(
            path, model=_plaque, device=DEVICE,
            plaque_threshold=THRESHOLD, plaque_min_area=MIN_AREA,
            wall_model_path=_wall_path,
            pixel_spacing_mm=float(pixel_spacing) if pixel_spacing else None,
            denoise="both",
            clinical={"age": int(age) if age is not None else None,
                      "sbp": sbp, "ldl": ldl,
                      "smoking": bool(smoking), "diabetes": bool(diabetes)})
    finally:
        if os.path.exists(path):
            os.unlink(path)

    annotated = res["annotated_image"]
    annotated_rgb = annotated[:, :, ::-1] if annotated is not None else None
    v = report.format_view(res, bool(clinician_reviewed))
    return annotated_rgb, v["summary"]


CSS = """
body, .gradio-container {
    background: #0f0c29;
    font-family: 'Inter', sans-serif;
}
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: .04em !important;
}
.gr-button-primary:hover {
    opacity: .88 !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(102,126,234,.5) !important;
}
"""

with gr.Blocks(title="TI-Artery — Carotid IMT & Stroke Risk", css=CSS,
               theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate")) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:28px 16px 8px;">
      <h1 style="font-size:2rem;font-weight:800;
                 background:linear-gradient(90deg,#818cf8,#c084fc,#fb7185);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 margin:0 0 8px;">
        TI-Artery
      </h1>
      <p style="color:#94a3b8;font-size:.95rem;margin:0;">
        Carotid B-mode · IMT measurement · Stroke-risk screening
      </p>
      <p style="color:#f87171;font-size:.8rem;margin:8px 0 0;">
        Research prototype — not a medical device. Every result must be reviewed by a qualified physician.
      </p>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ---- LEFT: inputs ----
        with gr.Column(scale=1, min_width=300):
            gr.HTML('<div style="color:#a5b4fc;font-size:.8rem;font-weight:700;'
                    'letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">'
                    'Upload B-mode Image</div>')
            image = gr.Image(
                label="Carotid B-mode still",
                type="numpy",
                elem_id="input-image",
            )
            gr.HTML('<div style="color:#a5b4fc;font-size:.8rem;font-weight:700;'
                    'letter-spacing:.08em;text-transform:uppercase;margin:14px 0 8px;">'
                    'Clinical Parameters (optional)</div>')
            with gr.Row():
                age = gr.Number(label="Age (years)", value=65, minimum=18, maximum=120)
                sbp = gr.Number(label="Systolic BP (mmHg)", value=130)
            with gr.Row():
                ldl = gr.Number(label="LDL (mg/dL)", value=110)
                spacing = gr.Number(label="Pixel spacing (mm/px)", value=None,
                                    info="From scanner — leave blank to use a demo estimate")
            with gr.Row():
                smoking = gr.Checkbox(label="Smoking")
                diabetes = gr.Checkbox(label="Diabetes")
            reviewed = gr.Checkbox(
                label="I am a clinician and will review this result",
                value=False)
            btn = gr.Button("▶  Analyze", variant="primary", size="lg")

        # ---- RIGHT: outputs ----
        with gr.Column(scale=1, min_width=340):
            gr.HTML('<div style="color:#a5b4fc;font-size:.8rem;font-weight:700;'
                    'letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">'
                    'Labeled Image</div>')
            out_img = gr.Image(label="Overlay (green=plaque, orange=wall)", elem_id="out-image")

            gr.HTML('<div style="color:#a5b4fc;font-size:.8rem;font-weight:700;'
                    'letter-spacing:.08em;text-transform:uppercase;margin:14px 0 8px;">'
                    'Analysis Report</div>')
            out_summary = gr.HTML(elem_id="out-summary")

    btn.click(
        analyze,
        inputs=[image, age, sbp, ldl, smoking, diabetes, spacing, reviewed],
        outputs=[out_img, out_summary],
    )

    gr.HTML("""
    <div style="text-align:center;padding:12px 16px 20px;
                color:#475569;font-size:.78rem;border-top:1px solid #1e293b;margin-top:24px;">
      Out of scope: hemodynamic stenosis (needs Doppler), intraplaque hemorrhage,
      3-D plaque volume, ICA/ECA/intracranial vessels.
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
