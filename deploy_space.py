"""Push the Gradio demo to Hugging Face Spaces without modifying training checkpoints."""
import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi, upload_folder


SPACE_README = """---
title: TI-Artery
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: 3.10.13
pinned: false
---

# TI-Artery

Research prototype for carotid IMT and plaque screening from B-mode ultrasound.
**Not a medical device.** A clinician must review every result.

Without valid scanner pixel spacing, image analysis and pixel measurements remain
available while calibrated measurements and the dependent risk assessment are
reported as unavailable. Assumed spacing is never used for risk scoring.

Scope: CCA B-mode screening only. No Doppler stenosis, no 3-D volume,
no vulnerable-plaque subtype, no automated treatment.
"""

SPACE_REQ = """# `spaces` is supplied by the Hugging Face ZeroGPU runtime; do not override its version.
--extra-index-url https://download.pytorch.org/whl/cu126
numpy==1.26.4
opencv-python-headless==4.11.0.86
pillow==12.1.1
gradio==6.31.0
torch==2.9.1
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-id", required=True,
        help="Existing Space ID in namespace/name format, e.g. username/ti-artery",
    )
    ap.add_argument("--plaque", default="models/plaque_finetune.pt")
    ap.add_argument("--wall", default="models/wall_unet.pt")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    plaque = args.plaque if os.path.isabs(args.plaque) else os.path.join(root, args.plaque)
    wall = args.wall if os.path.isabs(args.wall) else os.path.join(root, args.wall)
    for label, path in (("plaque", plaque), ("wall", wall)):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing required {label} model checkpoint: {path}")

    if args.repo_id.count("/") != 1 or any(not part for part in args.repo_id.split("/")):
        raise SystemExit("--repo-id must be a full namespace/name Space ID")

    # HfApi reads the user's existing cached/standard environment credentials.
    # Never print or manually inspect the token.
    api = HfApi()
    try:
        identity = api.whoami()
    except Exception as exc:
        raise SystemExit(
            "Hugging Face authentication is unavailable. Sign in with `hf auth login`, "
            "then rerun this command."
        ) from exc
    username = identity.get("name") if isinstance(identity, dict) else str(identity)
    if not username:
        raise SystemExit("Could not determine the authenticated Hugging Face account")
    if args.repo_id.split("/", 1)[0].casefold() != username.casefold():
        raise SystemExit(
            f"Authenticated account {username!r} does not own the requested Space "
            f"{args.repo_id!r}; no files were uploaded."
        )
    try:
        api.repo_info(repo_id=args.repo_id, repo_type="space")
    except Exception as exc:
        raise SystemExit(
            f"Could not access existing Space {args.repo_id!r}. Check the Space ID, "
            "permissions, and whether it already exists; this script will not create it."
        ) from exc
    runtime = api.get_space_runtime(repo_id=args.repo_id)
    hardware = getattr(runtime.hardware, "value", runtime.hardware)
    print(f"Target Space hardware: {hardware or 'not reported'} (no hardware changes requested)")
    with tempfile.TemporaryDirectory() as tmp:
        pkg = os.path.join(tmp, "space")
        os.makedirs(os.path.join(pkg, "tiartery"))
        os.makedirs(os.path.join(pkg, "models"))
        shutil.copy(os.path.join(root, "app.py"), os.path.join(pkg, "app.py"))
        for name in os.listdir(os.path.join(root, "tiartery")):
            if name.endswith(".py"):
                shutil.copy(os.path.join(root, "tiartery", name),
                            os.path.join(pkg, "tiartery", name))
        shutil.copy(plaque, os.path.join(pkg, "models", "plaque_finetune.pt"))
        shutil.copy(wall, os.path.join(pkg, "models", "wall_unet.pt"))
        with open(os.path.join(pkg, "README.md"), "w") as f:
            f.write(SPACE_README)
        with open(os.path.join(pkg, "requirements.txt"), "w") as f:
            f.write(SPACE_REQ)
        upload_folder(repo_id=args.repo_id, repo_type="space", folder_path=pkg)
    print(f"https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
