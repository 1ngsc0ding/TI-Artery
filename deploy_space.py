"""Push the Gradio demo to Hugging Face Spaces without modifying training checkpoints."""
import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi, create_repo, upload_folder


SPACE_README = """---
title: TI-Artery
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: 3.11
pinned: false
---

# TI-Artery

Research prototype for carotid IMT and plaque screening from B-mode ultrasound.
**Not a medical device.** A clinician must review every result.

Scope: CCA B-mode screening only. No Doppler stenosis, no 3-D volume,
no vulnerable-plaque subtype, no automated treatment.
"""

SPACE_REQ = """numpy>=1.24
opencv-python-headless>=4.8
pillow>=10.0
gradio==4.44.1
torch
"""


def main():
    ap = argparse.ArgumentParser()
    # A bare repo name is expanded to the authenticated user's namespace below.
    ap.add_argument("--repo-id", default=os.environ.get("TIARTERY_HF_REPO", "carotid-artery"))
    ap.add_argument("--plaque", default="models/plaque_finetune.pt")
    ap.add_argument("--wall", default="models/wall_unet.pt")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    plaque = args.plaque if os.path.isabs(args.plaque) else os.path.join(root, args.plaque)
    wall = args.wall if os.path.isabs(args.wall) else os.path.join(root, args.wall)
    if not os.path.exists(plaque):
        raise SystemExit(f"Missing plaque model: {plaque}")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    who = api.whoami()
    user = who["name"] if isinstance(who, dict) else who
    repo_id = args.repo_id
    if "/" not in repo_id:
        repo_id = f"{user}/{repo_id}"
    create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, token=token)
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
        if os.path.exists(wall):
            shutil.copy(wall, os.path.join(pkg, "models", "wall_unet.pt"))
        with open(os.path.join(pkg, "README.md"), "w") as f:
            f.write(SPACE_README)
        with open(os.path.join(pkg, "requirements.txt"), "w") as f:
            f.write(SPACE_REQ)
        upload_folder(repo_id=repo_id, repo_type="space", folder_path=pkg, token=token)
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
